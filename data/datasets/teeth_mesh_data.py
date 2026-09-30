"""Teeth mesh dataset.

Note: If you run this file directly (e.g. `python data/datasets/teeth_mesh_data.py`)
from outside the project root, Python may not be able to resolve the top-level
`data` package. The small block below makes direct execution more robust by
adding the repository root to `sys.path`.
"""

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import time
    from loguru import logger as log

import json
import os
import os.path as osp
import time
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from loguru import logger as log

from data.transforms.pre_transforms import compose_graph_pretransforms
from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData

LABEL_MAPPING = {
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    18: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}
LABEL_MAPPING.update({k + 20: v for k, v in LABEL_MAPPING.items()})
LABEL_MAPPING[0] = 0  # Background / unlabeled


@dataclass(frozen=True)
class TeethMeshId:
    """Identifier for a single jaw mesh.

    Expected format in split files: '<PATIENT>_<lower|upper>' (e.g., 'EJWZZZRF_lower').
    """

    patient_id: str
    tooth_type: str  # 'lower' | 'upper'

    @property
    def stem(self) -> str:
        return f"{self.patient_id}_{self.tooth_type}"


def _parse_mesh_id(mesh_id: str) -> TeethMeshId:
    mesh_id = mesh_id.strip()
    if not mesh_id:
        raise ValueError("Empty mesh id")

    if mesh_id.endswith(".obj"):
        mesh_id = mesh_id[: -len(".obj")]

    if "_" not in mesh_id:
        raise ValueError(
            f"Invalid teeth mesh id '{mesh_id}'. Expected '<PATIENT>_<lower|upper>'."
        )

    patient_id, tooth_type = mesh_id.rsplit("_", 1)
    if tooth_type not in {"lower", "upper"}:
        raise ValueError(
            f"Invalid teeth mesh id '{mesh_id}'. Suffix must be '_lower' or '_upper'."
        )

    return TeethMeshId(patient_id=patient_id, tooth_type=tooth_type)


def _triangulate_faces(face_indices: List[List[int]]) -> np.ndarray:
    """Triangulate faces using a fan if faces are polygons.

    Returns an array of shape (F, 3) with 0-based indices.
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
    """Build a bidirectional, coalesced edge_index from triangular faces.

    Produces both directions for each undirected edge.
    """

    if face_f3.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # face_f3: (3, F)
    a, b, c = face_f3[0], face_f3[1], face_f3[2]

    src = torch.cat([a, b, c, b, c, a], dim=0)
    dst = torch.cat([b, c, a, a, b, c], dim=0)

    # Remove self loops (shouldn't exist, but be safe)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # Coalesce / unique edges via linear key
    key = src.to(torch.int64) * int(num_nodes) + dst.to(torch.int64)
    key = torch.unique(key)
    src_u = (key // int(num_nodes)).to(torch.long)
    dst_u = (key % int(num_nodes)).to(torch.long)

    edge_index = torch.stack([src_u, dst_u], dim=0)
    return edge_index


def _simplify_mesh(
    vertices_fine: np.ndarray,
    faces_f3_fine: np.ndarray,
    target_faces: int,
    is_closed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplify mesh using PyMeshLab quadric decimation."""

    try:
        import pymeshlab as ml
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyMeshLab is required for mesh simplification. Install 'pymeshlab' or disable simplification."
        ) from e

    ms = ml.MeshSet()
    ms.add_mesh(
        ml.Mesh(
            vertex_matrix=vertices_fine.astype(np.float64),
            face_matrix=faces_f3_fine.astype(np.int32),
        ),
        "m",
    )

    ms.compute_normal_per_face()
    ms.compute_normal_per_vertex()

    preserve_boundary = not bool(is_closed)
    preserve_topology = True

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(max(4, target_faces)),
        preservenormal=True,
        preservetopology=preserve_topology,
        preserveboundary=preserve_boundary,
        optimalplacement=True,
        qualityweight=False,
        autoclean=True,
        planarquadric=True,
        selected=False,
    )

    m2 = ms.current_mesh()
    v2 = m2.vertex_matrix().astype(np.float64)
    f2 = m2.face_matrix().astype(np.int64)
    return v2, f2


def _map_vertex_labels_to_coarse(
    vertices_fine: np.ndarray,
    fine_labels: np.ndarray,
    vertices_coarse: np.ndarray,
) -> np.ndarray:
    """Map fine vertex labels onto coarse vertices with nearest-neighbour lookup."""

    try:
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SciPy is required for label mapping during simplification. Install 'scipy' or disable simplification."
        ) from e

    tree = cKDTree(vertices_fine)
    _, idx = tree.query(vertices_coarse, k=1, workers=1)
    idx = idx.astype(np.int64)
    return fine_labels[idx]


def _compute_face_labels_from_vertex_labels(
    face: torch.Tensor,
    vertex_labels: torch.Tensor,
) -> torch.Tensor:
    if face.numel() == 0:
        return torch.empty((0,), dtype=vertex_labels.dtype, device=vertex_labels.device)

    face_vertex_labels = vertex_labels[face]
    return face_vertex_labels[0].contiguous()


def _append_face_labels_to_data(data: Data) -> MeshData:
    mesh_data = MeshData.from_data(data)
    if not hasattr(mesh_data, "face") or mesh_data.face is None:
        raise ValueError("Cannot append face_labels to mesh without face connectivity")
    if not hasattr(mesh_data, "vertex_labels") or mesh_data.vertex_labels is None:
        raise ValueError("Cannot append face_labels to mesh without vertex_labels")

    mesh_data.face_labels = _compute_face_labels_from_vertex_labels(
        face=mesh_data.face,
        vertex_labels=mesh_data.vertex_labels.long(),
    )
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

    dataset = TeethMeshDataset(
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


class TeethMeshDataset(InMemoryDataset):
    """Teeth mesh dataset for jaw meshes stored as per-patient folders.

    Expected layout under ``root`` (typically '<dataroot>/all_data'):
      - <PATIENT>/<PATIENT>_lower.obj
      - <PATIENT>/<PATIENT>_lower.json
      - <PATIENT>/<PATIENT>_upper.obj
      - <PATIENT>/<PATIENT>_upper.json

    Split files contain mesh ids like '<PATIENT>_lower' (no extension).

    Output Data fields:
      - pos: (N, 3)
      - face: (3, F)
      - edge_index: (2, E)
      - vertex_labels: (N,)
            - face_labels: (F,)
      - filename: '<PATIENT>_<lower|upper>.obj' (for logging/compat)
      - mesh_id: '<PATIENT>_<lower|upper>'

        Notes:
            - By default, processed files are stored under a sibling directory of ``root``
                named ``processed_<split>``. You can override this by passing ``processed_dir=...``
                to load from (or write to) an explicit processed directory.
            - To support SLURM wall-time limits, you can optionally provide ``intermediate_dir``.
              When set, the dataset saves each processed ``MeshData`` as ``<mesh_id>.pt`` as soon
              as it is produced. If a job is interrupted, re-running with the same
              ``intermediate_dir`` will resume from the already-saved meshes.
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
        target_faces: Optional[int] = None,
        is_closed: bool = True,
        processed_dir: Optional[str] = None,
        intermediate_dir: Optional[str] = None,
        resume_from_intermediate: bool = True,
        force_reload: bool = False,
    ):
        self.split = split
        self.file_list = list(file_list) if file_list is not None else None
        self.profile_transforms = profile_transforms
        self.target_faces = target_faces
        self.is_closed = is_closed
        self._processed_dir_override = (
            osp.abspath(osp.expanduser(processed_dir)) if processed_dir else None
        )
        self._intermediate_dir_override = (
            osp.abspath(osp.expanduser(intermediate_dir)) if intermediate_dir else None
        )
        self.resume_from_intermediate = bool(resume_from_intermediate)

        if self.file_list is not None and pre_filter is None:
            allowed = {
                s.strip().removesuffix(".obj") for s in self.file_list if s.strip()
            }
            pre_filter = lambda data: getattr(data, "mesh_id", None) in allowed

        transform = (
            compose_graph_transforms(
                transforms or [], profile=profile_transforms, stage=split
            )
            if transforms is not None
            else None
        )

        pre_transform_names: Optional[List[str]] = pre_transforms

        pre_transform = (
            compose_graph_pretransforms(pre_transform_names)
            if pre_transform_names is not None
            else None
        )
        self._pre_transform_names = (
            list(pre_transform_names) if pre_transform_names else []
        )

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @staticmethod
    def _hash_list(items: List[str]) -> str:
        payload = "\n".join(items).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_intermediate_meta(self) -> dict:
        cleaned_file_list: Optional[List[str]]
        if self.file_list is None:
            cleaned_file_list = None
        else:
            cleaned_file_list = sorted(
                {
                    str(s).strip().removesuffix(".obj")
                    for s in self.file_list
                    if str(s).strip()
                }
            )

        meta = {
            "split": str(self.split),
            "target_faces": (
                int(self.target_faces) if self.target_faces is not None else None
            ),
            "is_closed": bool(self.is_closed),
            "pre_transforms": (
                list(self._pre_transform_names) if self._pre_transform_names else []
            ),
            "file_list_hash": (
                self._hash_list(cleaned_file_list)
                if cleaned_file_list is not None
                else None
            ),
        }
        meta["fingerprint"] = self._hash_list([json.dumps(meta, sort_keys=True)])
        return meta

    @property
    def processed_dir(self) -> str:
        if self._processed_dir_override is not None:
            return self._processed_dir_override
        return osp.join(os.path.dirname(self.root), f"new_processed_{self.split}")

    @property
    def intermediate_dir(self) -> Optional[str]:
        return self._intermediate_dir_override

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

        intermediate_dir = self.intermediate_dir
        intermediate_meta_path = (
            osp.join(intermediate_dir, "intermediate_meta.json")
            if intermediate_dir is not None
            else None
        )

        expected_meta = self._build_intermediate_meta() if intermediate_dir else None
        can_resume = (
            bool(self.resume_from_intermediate) and intermediate_dir is not None
        )

        if intermediate_dir is not None:
            os.makedirs(intermediate_dir, exist_ok=True)
            if expected_meta is not None:
                if osp.isfile(intermediate_meta_path):
                    try:
                        with open(intermediate_meta_path, "r") as f:
                            existing_meta = json.load(f)
                        if existing_meta.get("fingerprint") != expected_meta.get(
                            "fingerprint"
                        ):
                            raise RuntimeError(
                                "Intermediate cache metadata does not match current settings. "
                                "Use a different intermediate_dir or delete the existing one."
                            )
                    except RuntimeError:
                        raise
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to validate intermediate cache metadata at {intermediate_meta_path}: {e}"
                        )
                else:
                    try:
                        with open(intermediate_meta_path, "w") as f:
                            json.dump(expected_meta, f, indent=2, sort_keys=True)
                    except Exception as e:
                        log.warning(
                            f"Failed to write intermediate metadata to {intermediate_meta_path}: {e}"
                        )

        data_list: List[MeshData] = []
        signature_cache_dir = osp.join(self.processed_dir, "signature_spectra")
        for mesh_id_raw in mesh_ids:
            start = time.time()
            mesh_id_raw = str(mesh_id_raw).strip()
            if not mesh_id_raw:
                continue
            mesh_id = _parse_mesh_id(mesh_id_raw)

            cached_path = (
                osp.join(intermediate_dir, f"{mesh_id.stem}.pt")
                if intermediate_dir is not None
                else None
            )

            mesh_data: Optional[MeshData] = None
            if can_resume and cached_path is not None and osp.isfile(cached_path):
                try:
                    cached_obj = torch.load(cached_path, weights_only=False)
                    if isinstance(cached_obj, MeshData):
                        mesh_data = cached_obj
                    elif isinstance(cached_obj, Data):
                        mesh_data = MeshData.from_data(cached_obj)
                except Exception as e:
                    log.warning(f"Failed to load cached mesh {cached_path}: {e}")
                    mesh_data = None

            if mesh_data is None:
                obj_path = osp.join(
                    self.root, mesh_id.patient_id, f"{mesh_id.stem}.obj"
                )
                json_path = osp.join(
                    self.root, mesh_id.patient_id, f"{mesh_id.stem}.json"
                )

                data = self._load_pyg_data(
                    obj_path=obj_path,
                    label_path=json_path,
                    mesh_id=mesh_id.stem,
                    target_faces=self.target_faces,
                    is_closed=self.is_closed,
                )
                data.signature_cache_dir = signature_cache_dir

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                mesh_data = MeshData.from_data(data)

                if cached_path is not None:
                    try:
                        torch.save(mesh_data, cached_path)
                    except Exception as e:
                        log.warning(f"Failed to write cached mesh {cached_path}: {e}")

            data_list.append(mesh_data)
            end = time.time()
            log.info(f"Processed mesh {mesh_id.stem} in {end - start:.2f} seconds")

        if len(data_list) == 0:
            raise RuntimeError(
                f"Processed 0 meshes for split='{self.split}'. "
                "Check that split ids match '<PATIENT>_<lower|upper>' and files exist under root/all_data."
            )

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])

        if intermediate_meta_path is not None and expected_meta is not None:
            try:
                with open(intermediate_meta_path, "w") as f:
                    json.dump(expected_meta, f, indent=2, sort_keys=True)
            except Exception as e:
                log.warning(
                    f"Failed to write intermediate metadata to {intermediate_meta_path}: {e}"
                )

    def _discover_all_mesh_ids(self) -> List[str]:
        mesh_ids: List[str] = []
        if not osp.isdir(self.root):
            return mesh_ids

        for patient in sorted(os.listdir(self.root)):
            patient_dir = osp.join(self.root, patient)
            if not osp.isdir(patient_dir):
                continue
            for tooth_type in ("lower", "upper"):
                stem = f"{patient}_{tooth_type}"
                obj_path = osp.join(patient_dir, f"{stem}.obj")
                if osp.isfile(obj_path):
                    mesh_ids.append(stem)
        return mesh_ids

    def _load_pyg_data(
        self,
        obj_path: str,
        label_path: Optional[str],
        mesh_id: str,
        target_faces: Optional[int],
        is_closed: bool,
    ) -> Data:
        if not osp.isfile(obj_path):
            raise FileNotFoundError(f"Missing OBJ file: {obj_path}")

        vertices: List[List[float]] = []
        faces_raw: List[List[int]] = []

        with open(obj_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                if parts[0] == "v":
                    # Can be: v x y z [r g b]
                    if len(parts) < 4:
                        continue
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                elif parts[0] == "f":
                    # Faces can be v, v//vn, v/vt, v/vt/vn
                    idxs: List[int] = []
                    for token in parts[1:]:
                        if not token:
                            continue
                        idx = token.split("//")[0].split("/")[0]
                        if idx:
                            idxs.append(int(idx) - 1)  # OBJ is 1-based
                    if idxs:
                        faces_raw.append(idxs)

        vertices_fine = np.asarray(vertices, dtype=np.float64)
        faces_f3_fine = _triangulate_faces(faces_raw)

        if vertices_fine.shape[0] == 0:
            raise RuntimeError(f"No vertices found in OBJ: {obj_path}")

        if target_faces is not None and faces_f3_fine.shape[0] > 0:
            vertices_coarse, faces_f3 = _simplify_mesh(
                vertices_fine=vertices_fine,
                faces_f3_fine=faces_f3_fine,
                target_faces=int(target_faces),
                is_closed=is_closed,
            )
        else:
            vertices_coarse, faces_f3 = vertices_fine, faces_f3_fine

        pos = torch.tensor(vertices_coarse, dtype=torch.float32)
        if faces_f3.shape[0] > 0:
            face = torch.tensor(faces_f3, dtype=torch.long).t().contiguous()  # (3, F)
        else:
            face = torch.empty((3, 0), dtype=torch.long)

        # Center
        pos = pos - pos.mean(dim=0, keepdim=True)

        edge_index = _build_edge_index_from_faces(face, num_nodes=pos.shape[0])

        # Labels
        vertex_labels = torch.full((pos.shape[0],), -1, dtype=torch.long)
        if label_path is not None and osp.isfile(label_path):
            with open(label_path, "r") as f:
                label_data = json.load(f)
            if "labels" in label_data:
                label_data["labels"] = [
                    LABEL_MAPPING.get(label, label) for label in label_data["labels"]
                ]
                fine_labels = np.asarray(label_data["labels"], dtype=np.int64)
                if fine_labels.shape[0] != vertices_fine.shape[0]:
                    raise ValueError(
                        f"Label length mismatch for {label_path}: got {fine_labels.shape[0]} labels, "
                        f"expected {vertices_fine.shape[0]} (num vertices in OBJ)."
                    )

                if (
                    target_faces is not None
                    and vertices_coarse.shape[0] != vertices_fine.shape[0]
                ):
                    coarse_labels = _map_vertex_labels_to_coarse(
                        vertices_fine=vertices_fine,
                        fine_labels=fine_labels,
                        vertices_coarse=vertices_coarse,
                    )
                else:
                    coarse_labels = fine_labels

                if coarse_labels.shape[0] != pos.shape[0]:
                    raise ValueError(
                        f"Coarse label length mismatch for {label_path}: got {coarse_labels.shape[0]}, expected {pos.shape[0]}"
                    )
                vertex_labels = torch.tensor(coarse_labels, dtype=torch.long)

        data = Data(pos=pos, face=face, edge_index=edge_index)
        data.vertex_labels = vertex_labels
        data.face_labels = _compute_face_labels_from_vertex_labels(
            face=face,
            vertex_labels=vertex_labels,
        )
        data.mesh_id = mesh_id
        data.filename = f"{mesh_id}.obj"

        return data


if __name__ == "__main__":

    import argparse

    def open_ids_file(ids_file: Optional[str]) -> List[str]:
        if ids_file and os.path.isfile(ids_file):
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
    #   python -m data.datasets.teeth_mesh_data --data-dir /path/to/all_data --split train
    # Or load from an already-processed directory:
    #   python -m data.datasets.teeth_mesh_data --data-dir /path/to/all_data --split train \
    #     --processed-dir /path/to/processed_train
    parser = argparse.ArgumentParser(description="TeethMeshDataset quick smoke test")
    parser.add_argument(
        "--append-face-labels",
        action="store_true",
        help="Upgrade an existing processed cache by appending face_labels.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to Teeth3DS root directory 'all_data'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split name (affects processed directory naming).",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional path to a split ids file (one '<PATIENT>_<lower|upper>' per line).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Optional override: directory containing processed 'data.pt'.",
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
        "--intermediate-dir",
        type=str,
        default=None,
        help="Optional directory to save per-mesh intermediate MeshData checkpoints for resuming.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from intermediate_dir even if it contains cached meshes.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=16000,
        help="Optional simplification target number of faces.",
    )
    parser.add_argument(
        "--pre",
        type=str,
        default="normalized_coords,normals,on_manifold_face_info,face_adjacency,face_adjacency_edge_index",
        help="Comma-separated list of pre_transforms.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force (re)processing even if processed files exist.",
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

    data_dir = args.data_dir
    splits_dir = osp.join(osp.dirname(data_dir), "splits")
    file_ids = open_ids_file(f"{splits_dir}/{args.split}_ids.txt") if args.split != "all" else None

    print(
        f"Initializing TeethMeshDataset with root={data_dir}, split={args.split}, "
        f"{len(file_ids) if file_ids else 'all'} mesh ids, target_faces={args.target_faces}, "
        f"pre_transforms={args.pre}, processed_dir={args.processed_dir}, "
        f"intermediate_dir={args.intermediate_dir}, resume={not args.no_resume}, "
        f"force_reload={args.force_reload}"
    )

    dataset = TeethMeshDataset(
        root=data_dir,
        split=args.split,
        file_list=file_ids if file_ids else None,
        pre_transforms=_split_csv(args.pre),
        processed_dir=(_resolved_processed_dir_arg(args.processed_dir, args.split)),
        intermediate_dir=args.intermediate_dir,
        resume_from_intermediate=not args.no_resume,
        target_faces=args.target_faces,
        force_reload=args.force_reload,
    )

    print(f"Loaded {len(dataset)} meshes (split={args.split})")
    mesh = dataset[0]
    print(mesh)
