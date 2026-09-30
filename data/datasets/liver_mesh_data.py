"""Liver mesh dataset.

Note: If you run this file directly (e.g. `python data/datasets/liver_mesh_data.py`)
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

import argparse
import glob
import os
import os.path as osp
from typing import Any, Callable, Iterable, List, Optional, Set

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from data.transforms.pre_transforms import compose_graph_pretransforms
from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData

import time


class LiverMeshDataset(InMemoryDataset):
    """
    Dataset for loading .obj mesh files.

    Args:
        root (str): Root directory where the dataset should be saved.
        file_pattern (str): Glob pattern to match .obj files (e.g., '*.obj', 'mesh_*.obj')
        transform (callable, optional): A function/transform that takes in a Data object
            and returns a transformed version. Applied on-the-fly during access.
        pre_transform (callable, optional): A function/transform that takes in a Data object
            and returns a transformed version. Applied once during processing.
        pre_filter (callable, optional): A function that takes in a Data object and returns
            a boolean indicating whether to include it in the dataset.
    """

    def __init__(
        self,
        root: str,
        edges_root: str,
        labels_root: str,
        file_pattern: str = "*.obj",
        split: str = "train",
        file_list: Optional[List[str]] = None,
        transforms: Optional[List[Any]] = None,
        pre_transforms: Optional[List[str]] = None,
        pre_filter: Optional[Callable] = None,
        profile_transforms: bool = False,
        processed_dir: Optional[str] = None,
        force_reload: bool = False,
    ):
        # NOTE: Order: pre_filter -> pre_transform -> transform
        # NOTE: Order of aug/feat/pre_transform matters
        self.file_pattern = file_pattern
        self.labels_root = labels_root
        self.edges_root = edges_root
        self.split = split
        self.file_list = file_list
        self.profile_transforms = profile_transforms
        self._processed_dir_override = (
            osp.abspath(osp.expanduser(processed_dir)) if processed_dir else None
        )
        self._allowed_filenames = self._normalize_allowed_filenames(file_list)

        # If file_list is provided and no pre_filter, create a pre_filter from the list
        if (
            self._allowed_filenames is not None
            and pre_filter is None
            and self._processed_dir_override is None
        ):
            pre_filter = (
                lambda data: getattr(data, "filename", None) in self._allowed_filenames
            )

        if transforms is not None:
            transform = compose_graph_transforms(
                transforms, profile=profile_transforms, stage=split
            )
        else:
            transform = None
        if pre_transforms is not None:
            pre_transform = compose_graph_pretransforms(pre_transforms)
        else:
            pre_transform = None
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._apply_post_load_file_filter()

    @staticmethod
    def _normalize_allowed_filenames(
        file_list: Optional[Iterable[str]],
    ) -> Optional[Set[str]]:
        if file_list is None:
            return None

        allowed = set()
        for file_name in file_list:
            cleaned = str(file_name).strip()
            if not cleaned:
                continue
            allowed.add(cleaned)
            allowed.add(f"{cleaned.removesuffix('.obj')}.obj")
        return allowed or None

    def _apply_post_load_file_filter(self) -> None:
        if self._allowed_filenames is None or self._processed_dir_override is None:
            return

        filtered = []
        for index in range(len(self)):
            mesh = self.get(index)
            if getattr(mesh, "filename", None) in self._allowed_filenames:
                filtered.append(MeshData.from_data(mesh))

        if not filtered:
            raise RuntimeError(
                f"Loaded processed cache from '{self.processed_dir}' but found 0 meshes matching split='{self.split}'."
            )

        self.data, self.slices = self.collate(filtered)

    def get_transform_timing_report(self) -> str:
        """Get timing report from the transform if profiling is enabled."""
        if self.profile_transforms and hasattr(self.transform, "get_timing_report"):
            return self.transform.get_timing_report()
        else:
            return "Transform profiling is not enabled. Set profile_transforms=True when creating the dataset."

    def reset_transform_timing(self):
        """Reset timing statistics for the transform."""
        if self.profile_transforms and hasattr(self.transform, "reset_timing_stats"):
            self.transform.reset_timing_stats()

    @property
    def raw_file_names(self) -> List[str]:
        # Return list of .obj files in raw directory
        paths = glob.glob(osp.join(self.raw_dir, self.file_pattern))
        return [osp.basename(p) for p in sorted(paths)]

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self) -> str:
        # Include split name to avoid clashes between train/val/test
        return f"data.pt"

    @property
    def processed_dir(self) -> str:
        if self._processed_dir_override is not None:
            return self._processed_dir_override
        return osp.join(self.root, f"processed_{self.split}")

    def download(self):
        # Override if you need to download files
        pass

    def process(self):
        data_list = []
        signature_cache_dir = osp.join(self.processed_dir, "signature_spectra")

        for obj_file in self.raw_file_names:
            obj_path = osp.join(self.raw_dir, obj_file)
            label_path = osp.join(self.labels_root, obj_file.replace(".obj", ".eseg"))
            edge_path = osp.join(self.edges_root, obj_file.replace(".obj", ".edges"))
            start = time.time()
            data = self._load_pyg_data(obj_path, label_path, edge_path)
            data.signature_cache_dir = signature_cache_dir

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(MeshData.from_data(data))
            end = time.time()
            print(f"Processed {obj_file} in {end - start:.2f} seconds")

        if not data_list:
            raise RuntimeError(
                f"Processed 0 meshes for split='{self.split}'. Check your root path, split ids, and file pattern."
            )

        print(f"Processed {len(data_list)} {self.split} meshes.")
        print(f"Saving to {self.processed_paths[0]}")
        print(f"Example data object: {data_list[0]}")
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def _load_pyg_data(self, obj_path: str, label_path: str, edge_path: str) -> Data:
        """Load a single .obj file and return a Data object."""
        vertices = []
        normals = []
        faces = []
        labels = []

        # Extract filename for pre_filter usage
        obj_filename = osp.basename(obj_path)

        with open(obj_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                # Vertex positions
                if parts[0] == "v":
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                # Vertex normals
                elif parts[0] == "vn":
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])

                # Faces (format: vertex_idx//normal_idx or vertex_idx)
                elif parts[0] == "f":
                    face_vertices = []
                    for vertex_data in parts[1:]:
                        # Handle both 'v//vn' and 'v' formats
                        if "//" in vertex_data:
                            vertex_idx = int(vertex_data.split("//")[0])
                        elif "/" in vertex_data:
                            vertex_idx = int(vertex_data.split("/")[0])
                        else:
                            vertex_idx = int(vertex_data)
                        face_vertices.append(vertex_idx)
                    faces.append(face_vertices)

        # Convert to tensors
        pos = torch.tensor(vertices, dtype=torch.float)
        face = torch.tensor(faces, dtype=torch.long).t().contiguous()

        # Centralize
        pos = pos - pos.mean(dim=0, keepdim=True)

        # OBJ indices are 1-based, convert to 0-based
        face = face - 1

        # Edges
        edges = np.loadtxt(edge_path, dtype=int).T
        edge_labels = np.loadtxt(label_path, dtype=int) - 1  # Convert to 0-based

        assert edges.shape[0] == 2, f"Edge file {edge_path} should have 2 columns."
        assert (
            edges.shape[1] == edge_labels.shape[0]
        ), f"Number of edges and labels do not match in {edge_path} and {label_path}."

        # Create Data object
        data = Data(
            pos=pos,
            face=face,
            edge_index=torch.tensor(edges, dtype=torch.long),
            edge_labels=torch.tensor(edge_labels, dtype=torch.long),
        )

        # Add normals if they exist
        # Will be computed later
        # if normals:
        #     normal = torch.tensor(normals, dtype=torch.float)
        #     data.normal = normal

        # Store filename for pre_filter to access
        data.filename = obj_filename

        return data


if __name__ == "__main__":

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
        return [item.strip() for item in value.split(",") if item.strip()]

    parser = argparse.ArgumentParser(description="LiverMeshDataset quick smoke test")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to livermesh directory containing OBJ files.",
    )
    parser.add_argument(
        "--edges-dir",
        type=str,
        default=None,
        help="Optional path to edges directory. Defaults to sibling 'edges' directory.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Optional path to seg directory. Defaults to sibling 'seg' directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split name used for processed directory naming.",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional path to a split ids file (one mesh filename or stem per line).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Optional override: directory containing or receiving processed data.pt.",
    )
    parser.add_argument(
        "--pre",
        type=str,
        default="",
        help="Comma-separated list of pre_transforms.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="",
        help="Comma-separated list of runtime transforms.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.obj",
        help="Glob pattern for mesh files under data-dir.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force (re)processing even if processed files already exist.",
    )
    args = parser.parse_args()

    data_dir = osp.abspath(osp.expanduser(args.data_dir))
    parent_dir = osp.dirname(data_dir)
    edges_dir = (
        osp.abspath(osp.expanduser(args.edges_dir))
        if args.edges_dir is not None
        else osp.join(parent_dir, "edges")
    )
    labels_dir = (
        osp.abspath(osp.expanduser(args.labels_dir))
        if args.labels_dir is not None
        else osp.join(parent_dir, "seg")
    )
    file_ids = open_ids_file(args.ids_file)

    print(
        f"Initializing LiverMeshDataset with root={data_dir}, split={args.split}, "
        f"{len(file_ids) if file_ids else 'all'} mesh ids, pre_transforms={args.pre}, "
        f"transforms={args.transforms}, processed_dir={args.processed_dir}, "
        f"force_reload={args.force_reload}"
    )

    dataset = LiverMeshDataset(
        root=data_dir,
        edges_root=edges_dir,
        labels_root=labels_dir,
        file_pattern=args.file_pattern,
        split=args.split,
        file_list=file_ids if file_ids else None,
        transforms=_split_csv(args.transforms),
        pre_transforms=_split_csv(args.pre),
        processed_dir=args.processed_dir,
        force_reload=args.force_reload,
    )

    print(f"Loaded {len(dataset)} meshes (split={args.split})")
    mesh = dataset[0]
    print(mesh)
