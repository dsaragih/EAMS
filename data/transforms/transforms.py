from __future__ import annotations

import hashlib
import math
import random
import re
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch_geometric import data
import torch_geometric.transforms as T
from hydra.utils import instantiate

from data.types import MeshData
from loguru import logger

_HIERARCHY_POS_PATTERN = re.compile(r"^pos_\d+$")
_HIERARCHY_FACE_PATTERN = re.compile(r"^face_\d+$")
_HIERARCHY_MAP_PATTERN = re.compile(r"^fine_to_coarse_\d+$")
_VECTOR_FIELD_KEYS = {"pos", "normal", "norm"}
_NONRIGID_STALE_KEYS = {
    "normal",
    "norm",
    "hks",
    "face_area",
    "thickness",
    "thickness_paired_idx",
}


def _data_keys(data: MeshData) -> list[str]:
    return list(data.keys()) if hasattr(data, "keys") else []


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _sample_identifier(data: MeshData) -> str:
    for key in ("mesh_id", "filename"):
        if hasattr(data, key):
            value = getattr(data, key)
            if isinstance(value, str) and value:
                return value
    return f"num_nodes={int(getattr(data, 'num_nodes', len(data.pos)))}"


def _apply_linear_map(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[-1] != 3:
        raise ValueError(
            f"Expected a tensor of shape (N, 3), got shape={tuple(tensor.shape)}"
        )
    return tensor @ matrix.t()


def _iter_face_keys(data: MeshData) -> list[str]:
    return sorted(
        key
        for key in _data_keys(data)
        if key == "face" or _HIERARCHY_FACE_PATTERN.match(key)
    )


def _flip_face_winding(face: torch.Tensor) -> torch.Tensor:
    if face.ndim != 2:
        raise ValueError(
            f"Expected a face tensor with 2 dimensions, got {tuple(face.shape)}"
        )
    if face.shape[0] == 3:
        return face[[0, 2, 1], :]
    if face.shape[1] == 3:
        return face[:, [0, 2, 1]]
    raise ValueError(
        f"Expected a triangular face tensor with shape (3, F) or (F, 3), got {tuple(face.shape)}"
    )


def _is_orientation_reversing(matrix: torch.Tensor, eps: float = 1e-8) -> bool:
    det = torch.linalg.det(matrix.float())
    return bool(det < -eps)


def _iter_rigid_position_keys(data: MeshData) -> list[str]:
    return sorted(
        key
        for key in _data_keys(data)
        if key in {"pos", "pointcloud_pos"} or _HIERARCHY_POS_PATTERN.match(key)
    )


def _iter_rigid_vector_keys(data: MeshData) -> list[str]:
    return sorted(
        key
        for key in _data_keys(data)
        if key in {"normal", "norm", "pointcloud_normal"}
        or key.startswith("normal_")
        or key.startswith("norm_")
    )


def _nonrigid_stale_fields(data: MeshData) -> list[str]:
    stale_keys = []
    for key in _data_keys(data):
        if key in _NONRIGID_STALE_KEYS:
            stale_keys.append(key)
            continue
        if _HIERARCHY_POS_PATTERN.match(key):
            stale_keys.append(key)
            continue
        if _HIERARCHY_FACE_PATTERN.match(key):
            stale_keys.append(key)
            continue
        if _HIERARCHY_MAP_PATTERN.match(key):
            stale_keys.append(key)
    return stale_keys


def _raise_for_stale_geometry(
    data: MeshData,
    *,
    transform_name: str,
    stage: str,
    stale_fields: Sequence[str],
) -> None:
    if not stale_fields:
        return
    sample_id = _sample_identifier(data)
    stale_str = ", ".join(sorted(stale_fields))
    message = (
        f"{transform_name} cannot run during stage='{stage}' for sample '{sample_id}' "
        f"because it would invalidate cached geometry-dependent fields: {stale_str}. "
        "Remove those pre_transforms/features or add an explicit refresh path first."
    )
    logger.warning(message)
    raise RuntimeError(message)


_LABEL_ATTR_BY_KIND = {
    "vertex": "vertex_labels",
    "edge": "edge_labels",
    "face": "face_labels",
}



def _resolve_label_attr(label_kind: str, label_attr: str | None) -> str:
    if label_attr is not None:
        return str(label_attr)
    try:
        return _LABEL_ATTR_BY_KIND[label_kind]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported label_kind '{label_kind}'. Expected one of {sorted(_LABEL_ATTR_BY_KIND)}."
        ) from exc



def _get_label_tensor(
    data: MeshData,
    *,
    label_kind: str,
    label_attr: str | None,
    background_label: int,
) -> tuple[str, torch.Tensor]:
    resolved_attr = _resolve_label_attr(label_kind, label_attr)
    if not hasattr(data, resolved_attr):
        raise AttributeError(
            f"MorphLabels requires `data.{resolved_attr}` for label_kind='{label_kind}'."
        )

    labels = getattr(data, resolved_attr)
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"Expected `data.{resolved_attr}` to be a torch.Tensor, got {type(labels)!r}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"Expected `data.{resolved_attr}` to be 1D, got shape={tuple(labels.shape)}."
        )
    if torch.is_floating_point(labels) or torch.is_complex(labels):
        raise TypeError(
            f"`data.{resolved_attr}` must be an integer tensor, got dtype={labels.dtype}."
        )
    if int(background_label) < 0:
        raise ValueError("background_label must be a non-negative integer.")
    if labels.numel() > 0 and int(labels.min().item()) < 0:
        raise ValueError(
            f"`data.{resolved_attr}` must use non-negative integer labels, got min={int(labels.min().item())}."
        )
    return resolved_attr, labels



def _validate_index_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_width: int,
    expected_size: int,
) -> torch.Tensor:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 2
        or tensor.shape[1] != expected_width
    ):
        raise ValueError(
            f"Expected `{name}` with shape (M, {expected_width}), got {tuple(tensor.shape)}."
        )
    tensor = tensor.long()
    if tensor.numel() == 0:
        return tensor.reshape(0, expected_width)
    if int(tensor.min().item()) < 0 or int(tensor.max().item()) >= expected_size:
        raise ValueError(
            f"`{name}` contains ids that are incompatible with the chosen label tensor of size {expected_size}."
        )
    return tensor



def _build_bidirectional_pairs_from_undirected_edges(
    undirected_edges: torch.Tensor,
) -> torch.Tensor:
    if undirected_edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=undirected_edges.device)
    src = undirected_edges[:, 0]
    dst = undirected_edges[:, 1]
    directed = torch.stack(
        [torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)],
        dim=0,
    )
    return directed



def _build_vertex_adjacency_index(
    data: MeshData, *, expected_size: int
) -> torch.Tensor:
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise AttributeError("MorphLabels vertex mode requires `data.edge_index`.")
    edge_index = data.edge_index
    if (
        not isinstance(edge_index, torch.Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(
            f"Expected `data.edge_index` with shape (2, E), got {tuple(edge_index.shape)}."
        )
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    edge_index = edge_index.long()
    if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= expected_size:
        raise ValueError(
            "edge_index contains vertex ids that are incompatible with the chosen vertex label tensor."
        )

    undirected_edges = torch.sort(edge_index, dim=0).values.t().unique(dim=0)
    undirected_edges = undirected_edges[undirected_edges[:, 0] != undirected_edges[:, 1]]
    return _build_bidirectional_pairs_from_undirected_edges(undirected_edges)



def _build_edge_adjacency_index(data: MeshData, *, expected_size: int) -> torch.Tensor:
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise AttributeError("MorphLabels edge mode requires `data.edge_index`.")
    edge_index = data.edge_index
    if (
        not isinstance(edge_index, torch.Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(
            f"Expected `data.edge_index` with shape (2, E), got {tuple(edge_index.shape)}."
        )
    if int(edge_index.shape[1]) != expected_size:
        raise ValueError(
            "edge_labels must align with the current `data.edge_index` columns for MorphLabels edge mode: "
            f"got labels={expected_size}, edges={int(edge_index.shape[1])}."
        )

    vertex2edge = getattr(data, "vertex2edge", None)
    if vertex2edge is None:
        edge_indices = torch.arange(expected_size, device=edge_index.device)
        vertex2edge = torch.stack(
            [
                torch.cat([edge_index[0], edge_index[1]]),
                torch.cat([edge_indices, edge_indices]),
            ],
            dim=1,
        )

    vertex2edge = _validate_index_tensor(
        vertex2edge,
        name="data.vertex2edge",
        expected_width=2,
        expected_size=expected_size,
    )
    if vertex2edge.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    # Build a sparse vertex-edge incidence matrix, then use H^T H to recover
    # all edge pairs that share at least one vertex.
    vertex2edge = torch.unique(vertex2edge, dim=0)
    num_vertices = int(vertex2edge[:, 0].max().item()) + 1
    incidence_index = vertex2edge.t().contiguous()
    incidence_values = torch.ones(
        incidence_index.shape[1], device=edge_index.device, dtype=torch.float32
    )
    incidence = torch.sparse_coo_tensor(
        incidence_index,
        incidence_values,
        size=(num_vertices, expected_size),
        device=edge_index.device,
    ).coalesce()

    adjacency = torch.sparse.mm(incidence.transpose(0, 1), incidence).coalesce()
    row, col = adjacency.indices()
    values = adjacency.values()
    mask = (row != col) & (values > 0)
    return torch.stack([row[mask], col[mask]], dim=0)


def _build_face_adjacency_index(data: MeshData, *, expected_size: int) -> torch.Tensor:
    face_adjacency = getattr(data, "face_adjacency", None)
    if face_adjacency is None:
        raise AttributeError(
            "MorphLabels face mode requires cached `data.face_adjacency`. "
            "Add the `face_adjacency` pre-transform first."
        )

    face_adjacency = _validate_index_tensor(
        face_adjacency,
        name="data.face_adjacency",
        expected_width=2,
        expected_size=expected_size,
    )
    if face_adjacency.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=face_adjacency.device)

    face_adjacency = face_adjacency[face_adjacency[:, 0] != face_adjacency[:, 1]]
    return _build_bidirectional_pairs_from_undirected_edges(face_adjacency)



def _build_label_adjacency_index(
    data: MeshData, *, label_kind: str, expected_size: int
) -> torch.Tensor:
    if label_kind == "vertex":
        return _build_vertex_adjacency_index(data, expected_size=expected_size)
    if label_kind == "edge":
        return _build_edge_adjacency_index(data, expected_size=expected_size)
    if label_kind == "face":
        return _build_face_adjacency_index(data, expected_size=expected_size)
    raise ValueError(
        f"Unsupported label_kind '{label_kind}'. Expected one of {sorted(_LABEL_ATTR_BY_KIND)}."
    )



def _morph_dilate(
    labels: torch.Tensor, adjacency_index: torch.Tensor, background_label: int
) -> torch.Tensor:
    if adjacency_index.numel() == 0 or labels.numel() == 0:
        return labels.clone()

    target_idx = adjacency_index[0]
    neighbor_idx = adjacency_index[1]
    target_labels = labels[target_idx]
    neighbor_labels = labels[neighbor_idx]

    claim_mask = (target_labels == background_label) & (neighbor_labels != background_label)
    if not bool(claim_mask.any()):
        return labels.clone()

    claimed_targets = target_idx[claim_mask]
    claimed_labels = neighbor_labels[claim_mask].long()
    num_targets = int(labels.shape[0])
    num_classes = max(int(labels.max().item()), int(background_label)) + 1

    encoded = claimed_targets * num_classes + claimed_labels
    unique_encoded, claim_counts = torch.unique(encoded, sorted=True, return_counts=True)
    unique_targets = torch.div(unique_encoded, num_classes, rounding_mode="floor")
    unique_labels = torch.remainder(unique_encoded, num_classes)

    max_counts = torch.zeros(num_targets, dtype=claim_counts.dtype, device=labels.device)
    max_counts.scatter_reduce_(0, unique_targets, claim_counts, reduce="amax", include_self=True)

    winning_mask = claim_counts == max_counts[unique_targets]
    winning_targets = unique_targets[winning_mask]
    winning_labels = unique_labels[winning_mask]

    sentinel = torch.full((num_targets,), num_classes, dtype=winning_labels.dtype, device=labels.device)
    sentinel.scatter_reduce_(0, winning_targets, winning_labels, reduce="amin", include_self=True)

    updated = labels.clone()
    assign_mask = sentinel != num_classes
    updated[assign_mask] = sentinel[assign_mask].to(dtype=labels.dtype)
    return updated



def _morph_erode(
    labels: torch.Tensor, adjacency_index: torch.Tensor, background_label: int
) -> torch.Tensor:
    if adjacency_index.numel() == 0 or labels.numel() == 0:
        return labels.clone()

    target_idx = adjacency_index[0]
    neighbor_idx = adjacency_index[1]
    target_labels = labels[target_idx]
    boundary_mask = (target_labels != background_label) & (labels[neighbor_idx] != target_labels)
    if not bool(boundary_mask.any()):
        return labels.clone()

    boundary_targets = torch.unique(target_idx[boundary_mask])
    updated = labels.clone()
    updated[boundary_targets] = background_label
    return updated


def _axis_to_vector(axis: str | Sequence[float]) -> torch.Tensor:
    if isinstance(axis, str):
        axis_map = {
            "x": torch.tensor([1.0, 0.0, 0.0]),
            "y": torch.tensor([0.0, 1.0, 0.0]),
            "z": torch.tensor([0.0, 0.0, 1.0]),
            "random": torch.randn(3),
        }
        if axis not in axis_map:
            raise ValueError(
                f"Unsupported axis '{axis}'. Expected one of x, y, z, random."
            )
        return axis_map[axis]
    vector = torch.as_tensor(axis, dtype=torch.float32)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3D axis vector, got shape={tuple(vector.shape)}")
    norm = torch.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("Axis vector must be non-zero.")
    return vector / norm


def _sample_uniform_range(
    bounds: Sequence[float], *, name: str, rng: random.Random | None = None
) -> float:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {bounds!r}")
    low = float(bounds[0])
    high = float(bounds[1])
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound, got {bounds!r}")
    sampler = random.uniform if rng is None else rng.uniform
    return float(sampler(low, high))


def _random_unit_vector(
    eps: float = 1e-12, rng: random.Random | None = None
) -> torch.Tensor:
    if rng is None:
        axis = torch.randn(3, dtype=torch.float32)
    else:
        axis = torch.tensor(
            [rng.gauss(0.0, 1.0) for _ in range(3)], dtype=torch.float32
        )
    norm = torch.linalg.norm(axis)
    while norm <= eps:
        if rng is None:
            axis = torch.randn(3, dtype=torch.float32)
        else:
            axis = torch.tensor(
                [rng.gauss(0.0, 1.0) for _ in range(3)], dtype=torch.float32
            )
        norm = torch.linalg.norm(axis)
    return axis / norm


def _rodrigues_rotation_matrix(
    axis: str | Sequence[float], degrees: float
) -> torch.Tensor:
    axis_vector = _axis_to_vector(axis)
    theta = math.radians(float(degrees))
    kx, ky, kz = axis_vector.tolist()
    skew = torch.tensor(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=torch.float32,
    )
    identity = torch.eye(3, dtype=torch.float32)
    return identity + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _rotation_matrix(degrees: float, axis: str | Sequence[float]) -> torch.Tensor:
    return _rodrigues_rotation_matrix(axis=axis, degrees=degrees)


def _reflection_matrix(axis: str | Sequence[float]) -> torch.Tensor:
    normal = _axis_to_vector(axis)
    identity = torch.eye(3, dtype=torch.float32)
    outer = torch.outer(normal, normal)
    return identity - 2.0 * outer


def _apply_rigid_transform(
    data: MeshData,
    *,
    matrix: torch.Tensor,
    translation: torch.Tensor | None = None,
) -> MeshData:
    matrix = matrix.to(device=data.pos.device, dtype=data.pos.dtype)
    orientation_reversing = _is_orientation_reversing(matrix)
    translation_value = None
    if translation is not None:
        translation_value = translation.to(device=data.pos.device, dtype=data.pos.dtype)

    for key in _iter_rigid_position_keys(data):
        value = getattr(data, key)
        if isinstance(value, torch.Tensor):
            transformed = _apply_linear_map(value, matrix)
            if translation_value is not None:
                transformed = transformed + translation_value
            setattr(data, key, transformed)

    for key in _iter_rigid_vector_keys(data):
        value = getattr(data, key)
        if isinstance(value, torch.Tensor):
            setattr(data, key, _apply_linear_map(value, matrix))

    if orientation_reversing:
        for key in _iter_face_keys(data):
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                setattr(data, key, _flip_face_winding(value))
    return data


class TimedCompose(T.Compose):
    """Compose transform that measures execution time for each transform."""

    def __init__(self, transforms, log_times=True):
        super().__init__(transforms)
        self.log_times = log_times
        self.timing_stats = {}

    def __call__(self, data):
        for transform in self.transforms:
            transform_name = (
                transform.__class__.__name__
                if hasattr(transform, "__class__")
                else str(transform)
            )
            if hasattr(transform, "__name__"):
                transform_name = transform.__name__

            start_time = time.perf_counter()
            data = transform(data)
            elapsed_time = time.perf_counter() - start_time

            # Accumulate timing stats
            if transform_name not in self.timing_stats:
                self.timing_stats[transform_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "max_time": 0.0,
                }

            self.timing_stats[transform_name]["count"] += 1
            self.timing_stats[transform_name]["total_time"] += elapsed_time
            self.timing_stats[transform_name]["max_time"] = max(
                self.timing_stats[transform_name]["max_time"], elapsed_time
            )

            if self.log_times:
                logger.debug(f"{transform_name}: {elapsed_time*1000:.2f}ms")

        return data

    def get_timing_report(self):
        """Generate a timing report for all transforms."""
        report = "\n=== Transform Timing Report ===\n"
        report += f"{'Transform':<30} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>12} {'Max(ms)':>12}\n"
        report += "-" * 77 + "\n"

        # Sort by total time descending
        sorted_stats = sorted(
            self.timing_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
        )

        for transform_name, stats in sorted_stats:
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            report += f"{transform_name:<30} {stats['count']:>8} {stats['total_time']*1000:>12.2f} {avg_time*1000:>12.2f} {stats['max_time']*1000:>12.2f}\n"

        total_time = sum(s["total_time"] for s in self.timing_stats.values())
        report += "-" * 77 + "\n"
        report += f"{'TOTAL':<30} {'':<8} {total_time*1000:>12.2f}\n"

        return report

    def reset_timing_stats(self):
        """Reset all timing statistics."""
        self.timing_stats = {}


def compose_graph_transforms(
    transforms: Sequence[Any], profile: bool = False, stage: str = "train"
) -> T.Compose:
    """Compose runtime graph transforms from structured Hydra transform specs."""
    callable_list = []

    for transform_spec in transforms:
        transform = _instantiate_transform(transform_spec)
        _validate_transform_for_stage(transform, stage=stage)
        callable_list.append(_bind_transform_stage(transform, stage=stage))

    if profile:
        return TimedCompose(callable_list, log_times=True)
    return T.Compose(callable_list)


def _instantiate_transform(transform_spec: Any):
    if callable(transform_spec):
        return transform_spec
    if isinstance(transform_spec, str):
        raise ValueError(
            "Runtime transforms must now be configured as structured Hydra specs with _target_. "
            f"Received legacy string spec: {transform_spec}"
        )
    transform = instantiate(transform_spec)
    if not callable(transform):
        raise TypeError(
            f"Instantiated runtime transform must be callable, got {type(transform)!r}"
        )
    return transform


def _validate_transform_for_stage(transform: Any, *, stage: str) -> None:
    if stage in {"val", "test"} and not getattr(transform, "deterministic", True):
        raise ValueError(
            f"Transform {transform!r} is not deterministic and cannot be used for stage='{stage}'."
        )


def _bind_transform_stage(transform: Any, *, stage: str):
    def _apply(data: MeshData):
        if hasattr(transform, "set_stage"):
            transform.set_stage(stage)
        elif hasattr(transform, "stage"):
            transform.stage = stage
        return transform(data)

    _apply.__name__ = getattr(transform, "__class__", type(transform)).__name__
    return _apply


class RuntimeMeshTransform:
    deterministic: bool = True
    geometry_mode: str = "none"

    def __init__(self):
        self.stage = "train"

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def __call__(self, data: MeshData) -> MeshData:
        return self.apply(data)

    def apply(self, data: MeshData) -> MeshData:
        raise NotImplementedError


class BinarizeLabels(RuntimeMeshTransform):
    deterministic = True
    geometry_mode = "none"

    def apply(self, data: MeshData) -> MeshData:
        if not hasattr(data, "vertex_labels"):
            raise AttributeError("BinarizeLabels requires `data.vertex_labels`.")
        data.vertex_labels = (data.vertex_labels != 0).long()
        return data


class IgnoreClassLabels(RuntimeMeshTransform):
    """Replace selected labels with the loss and metric ignore index."""

    deterministic = True
    geometry_mode = "none"

    def __init__(self, class_ids: Iterable[int], ignore_index: int = -1):
        super().__init__()
        self.class_ids = tuple(sorted({int(class_id) for class_id in class_ids}))
        if any(class_id < 0 for class_id in self.class_ids):
            raise ValueError("class_ids must be non-negative integers.")
        self.ignore_index = int(ignore_index)

    def apply(self, data: MeshData) -> MeshData:
        for attr in ("vertex_labels", "face_labels"):
            labels = getattr(data, attr, None)
            if labels is None:
                continue
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"data.{attr} must be a torch.Tensor.")
            if self.class_ids:
                data[attr] = labels.masked_fill(
                    torch.isin(labels, labels.new_tensor(self.class_ids)),
                    self.ignore_index,
                )
        return data


class MorphLabels(RuntimeMeshTransform):
    """Apply graph-based morphology to vertex, edge, or face labels.

    Required cached topology:
    - `label_kind="vertex"` uses `data.edge_index`.
    - `label_kind="edge"` uses `data.vertex2edge` when available, otherwise falls back to `data.edge_index`.
    - `label_kind="face"` requires `data.face_adjacency`.
    """

    geometry_mode = "none"

    def __init__(
        self,
        mode: str,
        label_kind: str,
        label_attr: str | None = None,
        iterations: int = 1,
        background_label: int = 0,
        apply_prob: float = 1.0,
    ):
        super().__init__()
        if mode not in {"dilate", "erode"}:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of {{'dilate', 'erode'}}."
            )
        if label_kind not in _LABEL_ATTR_BY_KIND:
            raise ValueError(
                f"Unsupported label_kind '{label_kind}'. Expected one of {sorted(_LABEL_ATTR_BY_KIND)}."
            )
        if int(iterations) < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}.")
        if not 0.0 <= float(apply_prob) <= 1.0:
            raise ValueError(f"apply_prob must be in [0, 1], got {apply_prob}.")

        self.mode = mode
        self.label_kind = label_kind
        self.label_attr = label_attr
        self.iterations = int(iterations)
        self.background_label = int(background_label)
        self.apply_prob = float(apply_prob)
        self.deterministic = self.apply_prob >= 1.0

    def apply(self, data: MeshData) -> MeshData:
        if (
            self.stage == "train"
            and self.apply_prob < 1.0
            and random.random() > self.apply_prob
        ):
            return data

        resolved_attr, labels = _get_label_tensor(
            data,
            label_kind=self.label_kind,
            label_attr=self.label_attr,
            background_label=self.background_label,
        )
        adjacency_index = _build_label_adjacency_index(
            data, label_kind=self.label_kind, expected_size=int(labels.shape[0])
        )

        updated = labels.clone()
        morph_fn = _morph_dilate if self.mode == "dilate" else _morph_erode
        for _ in range(self.iterations):
            updated = morph_fn(
                updated,
                adjacency_index=adjacency_index,
                background_label=self.background_label,
            )

        setattr(data, resolved_attr, updated)
        return data


class LegacyTGCNNormalize(RuntimeMeshTransform):
    deterministic = True
    geometry_mode = "none"

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def apply(self, data: MeshData) -> MeshData:
        if not hasattr(data, "pos") or data.pos is None:
            raise AttributeError("LegacyTGCNNormalize requires `data.pos`.")
        if not hasattr(data, "face") or data.face is None:
            raise AttributeError("LegacyTGCNNormalize requires `data.face`.")

        source_normals = getattr(data, "norm", None)
        if source_normals is None:
            source_normals = getattr(data, "normal", None)
        if source_normals is None:
            raise AttributeError(
                "LegacyTGCNNormalize requires `data.norm` or `data.normal`."
            )

        face = data.face.long().t().contiguous()
        num_faces = int(face.shape[0])
        dtype = data.pos.dtype
        device = data.pos.device

        if num_faces == 0:
            data.tcgn_face_coor = torch.empty((0, 12), dtype=dtype, device=device)
            data.tcgn_face_normal = torch.empty((0, 12), dtype=dtype, device=device)
            return data

        pos_np = data.pos.detach().cpu().to(dtype=torch.float32).numpy()
        normals_np = source_normals.detach().cpu().to(dtype=torch.float32).numpy()
        face_np = face.detach().cpu().numpy()

        xyz_face = np.concatenate(
            (
                pos_np[face_np[:, 0], :],
                pos_np[face_np[:, 1], :],
                pos_np[face_np[:, 2], :],
            ),
            axis=1,
        )
        normal_vertex = np.concatenate(
            (
                normals_np[face_np[:, 0], :],
                normals_np[face_np[:, 1], :],
                normals_np[face_np[:, 2], :],
            ),
            axis=1,
        )

        xy = xyz_face[:, :3] - xyz_face[:, 3:6]
        xz = xyz_face[:, :3] - xyz_face[:, 6:9]
        normal_face = np.stack(
            [
                xy[:, 1] * xz[:, 2] - xz[:, 1] * xy[:, 2],
                xy[:, 2] * xz[:, 0] - xz[:, 2] * xy[:, 0],
                xy[:, 0] * xz[:, 1] - xz[:, 0] * xy[:, 1],
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        x_centre = (xyz_face[:, 0] + xyz_face[:, 3] + xyz_face[:, 6]) / 3
        y_centre = (xyz_face[:, 1] + xyz_face[:, 4] + xyz_face[:, 7]) / 3
        z_centre = (xyz_face[:, 2] + xyz_face[:, 5] + xyz_face[:, 8]) / 3
        centre_face = np.concatenate(
            (
                x_centre.reshape(num_faces, 1),
                y_centre.reshape(num_faces, 1),
                z_centre.reshape(num_faces, 1),
            ),
            axis=1,
        ).astype(np.float32, copy=False)

        points_face = np.concatenate(
            (xyz_face, centre_face, normal_vertex, normal_face), axis=1
        ).astype(np.float32, copy=False)
        points = np.concatenate((pos_np.copy(), normals_np.copy()), axis=1).astype(
            np.float32, copy=False
        )

        aug_translation = getattr(data, "tcgn_aug_translation", None)
        aug_theta = getattr(data, "tcgn_aug_theta", None)
        if aug_translation is not None and aug_theta is not None:
            x_bias, y_bias, z_bias = [float(v) for v in aug_translation]
            theta = float(aug_theta)

            points_face[:, 0] = points_face[:, 0] + x_bias
            points_face[:, 3] = points_face[:, 3] + x_bias
            points_face[:, 6] = points_face[:, 6] + x_bias
            points_face[:, 9] = points_face[:, 9] + x_bias
            points_face[:, 12] = points_face[:, 12] + x_bias
            points_face[:, 15] = points_face[:, 15] + x_bias
            points_face[:, 18] = points_face[:, 18] + x_bias
            points_face[:, 21] = points_face[:, 21] + x_bias

            points_face[:, 1] = points_face[:, 1] + y_bias
            points_face[:, 4] = points_face[:, 4] + y_bias
            points_face[:, 7] = points_face[:, 7] + y_bias
            points_face[:, 10] = points_face[:, 10] + y_bias
            points_face[:, 13] = points_face[:, 13] + y_bias
            points_face[:, 16] = points_face[:, 16] + y_bias
            points_face[:, 19] = points_face[:, 19] + y_bias
            points_face[:, 22] = points_face[:, 22] + y_bias

            points_face[:, 2] = points_face[:, 2] + z_bias
            points_face[:, 5] = points_face[:, 5] + z_bias
            points_face[:, 8] = points_face[:, 8] + z_bias
            points_face[:, 11] = points_face[:, 11] + z_bias
            points_face[:, 14] = points_face[:, 14] + z_bias
            points_face[:, 17] = points_face[:, 17] + z_bias
            points_face[:, 20] = points_face[:, 20] + z_bias
            points_face[:, 23] = points_face[:, 23] + z_bias

            rotation_x = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(theta), -math.sin(theta)],
                    [0.0, math.sin(theta), math.cos(theta)],
                ],
                dtype=np.float32,
            )
            rotation_y = np.array(
                [
                    [math.cos(theta), 0.0, math.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(theta), 0.0, math.cos(theta)],
                ],
                dtype=np.float32,
            )
            rotation_z = np.array(
                [
                    [math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            for start in range(0, 24, 3):
                points_face[:, start : start + 3] = points_face[
                    :, start : start + 3
                ].dot(rotation_x.transpose())
            for start in range(0, 24, 3):
                points_face[:, start : start + 3] = points_face[
                    :, start : start + 3
                ].dot(rotation_y.transpose())
            for start in range(0, 24, 3):
                points_face[:, start : start + 3] = points_face[
                    :, start : start + 3
                ].dot(rotation_z.transpose())

        centre = points_face[:, 9:12].mean(axis=0)
        points_face[:, 0:3] -= centre
        points_face[:, 3:6] -= centre
        points_face[:, 6:9] -= centre
        points_face[:, 9:12] = (
            points_face[:, 0:3] + points_face[:, 3:6] + points_face[:, 6:9]
        ) / 3
        points[:, :3] -= centre

        scale = max(float(points.max()), self.eps)
        points_face[:, :12] = points_face[:, :12] / scale

        maxs = points[:, :3].max(axis=0)
        mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = np.maximum(points[:, :3].std(axis=0), self.eps)
        nmeans = points[:, 3:].mean(axis=0)
        nstds = np.maximum(points[:, 3:].std(axis=0), self.eps)
        nmeans_f = points_face[:, 21:].mean(axis=0)
        nstds_f = np.maximum(points_face[:, 21:].std(axis=0), self.eps)

        for i in range(3):
            points_face[:, i] = (points_face[:, i] - means[i]) / stds[i]
            points_face[:, i + 3] = (points_face[:, i + 3] - means[i]) / stds[i]
            points_face[:, i + 6] = (points_face[:, i + 6] - means[i]) / stds[i]
            points_face[:, i + 9] = (points_face[:, i + 9] - mins[i]) / (
                max(maxs[i] - mins[i], self.eps)
            )
            points_face[:, i + 12] = (points_face[:, i + 12] - nmeans[i]) / nstds[i]
            points_face[:, i + 15] = (points_face[:, i + 15] - nmeans[i]) / nstds[i]
            points_face[:, i + 18] = (points_face[:, i + 18] - nmeans[i]) / nstds[i]
            points_face[:, i + 21] = (points_face[:, i + 21] - nmeans_f[i]) / nstds_f[i]

        tcgn_face = torch.from_numpy(points_face).to(device=device, dtype=dtype)
        data.tcgn_face_coor = tcgn_face[:, :12]
        data.tcgn_face_normal = tcgn_face[:, 12:24]
        return data


class NormalizePositions(RuntimeMeshTransform):
    deterministic = True
    geometry_mode = "none"

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def apply(self, data: MeshData) -> MeshData:
        if not hasattr(data, "pos") or data.pos is None:
            raise AttributeError("NormalizePositions requires `data.pos`.")
        pos = data.pos.float()
        if pos.ndim == 2 and pos.shape[1] == 3:
            center = pos.mean(dim=0, keepdim=True)
            normalized = pos - center
            scale = normalized.norm(dim=-1).max().clamp_min(self.eps)
            data.pos = normalized / scale
            return data
        raise ValueError(f"Unsupported pos shape for normalization: {tuple(pos.shape)}")


class PointCloudSample(RuntimeMeshTransform):
    deterministic = True
    geometry_mode = "pointcloud"

    def __init__(
        self,
        n_points: int = 1024,
        sample_mode_train: str = "random",
        sample_mode_eval: str = "farthest",
        sample_with_replacement_train: bool = False,
        sample_with_replacement_eval: bool = False,
        normalize: bool = True,
        scale_low: float = 2.0 / 3.0,
        scale_high: float = 3.0 / 2.0,
        shift_range: float = 0.2,
        include_normals: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        if n_points < 1:
            raise ValueError(f"n_points must be positive, got: {n_points}")
        if sample_mode_train not in {"random", "farthest"}:
            raise ValueError("sample_mode_train must be one of {'random', 'farthest'}")
        if sample_mode_eval not in {"random", "farthest"}:
            raise ValueError("sample_mode_eval must be one of {'random', 'farthest'}")

        self.n_points = int(n_points)
        self.sample_mode_train = sample_mode_train
        self.sample_mode_eval = sample_mode_eval
        self.sample_with_replacement_train = bool(sample_with_replacement_train)
        self.sample_with_replacement_eval = bool(sample_with_replacement_eval)
        self.normalize = bool(normalize)
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        self.shift_range = float(shift_range)
        self.include_normals = bool(include_normals)
        self.eps = float(eps)

    def set_stage(self, stage: str) -> None:
        super().set_stage(stage)
        self.deterministic = stage in {"val", "test"}

    def _sample_indices(self, pos: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_nodes = int(pos.shape[0])
        if num_nodes < 1:
            raise ValueError("PointCloudSample cannot sample from an empty point set")

        sample_with_replacement = (
            self.sample_with_replacement_train
            if self.stage == "train"
            else self.sample_with_replacement_eval
        )
        if sample_with_replacement:
            indices = torch.randint(num_nodes, (self.n_points,), device=pos.device)
            return indices, self.n_points

        sample_count = min(self.n_points, num_nodes)
        mode = (
            self.sample_mode_train if self.stage == "train" else self.sample_mode_eval
        )

        if mode == "random":
            indices = torch.randperm(num_nodes, device=pos.device)[:sample_count]
        else:
            from models.layers.point_conv import farthest_point_sample

            indices = farthest_point_sample(pos.unsqueeze(0), sample_count).squeeze(0)

        if sample_count == self.n_points:
            return indices, sample_count

        pad_count = self.n_points - sample_count
        if self.stage == "train":
            pad_indices = indices[
                torch.randint(sample_count, (pad_count,), device=pos.device)
            ]
        else:
            pad_indices = indices[:1].repeat(pad_count)
        return torch.cat([indices, pad_indices], dim=0), sample_count

    def _normalize_points(
        self, points: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize:
            return points
        valid_points = points[valid_mask]
        if valid_points.numel() == 0:
            return points
        center = valid_points.mean(dim=0, keepdim=True)
        normalized = points - center
        scale = valid_points.sub(center).norm(dim=-1).amax().clamp_min(self.eps)
        return normalized / scale

    def _augment_points(
        self, points: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.stage != "train":
            return points
        scale = torch.empty(1, device=points.device, dtype=points.dtype).uniform_(
            self.scale_low, self.scale_high
        )
        shift = torch.empty(3, device=points.device, dtype=points.dtype).uniform_(
            -self.shift_range, self.shift_range
        )
        augmented = points.clone()
        augmented[valid_mask] = augmented[valid_mask] * scale + shift
        return augmented

    def apply(self, data: MeshData) -> MeshData:
        if not hasattr(data, "pos") or data.pos is None:
            raise AttributeError("PointCloudSample requires `data.pos`.")
        if self.include_normals and (
            not hasattr(data, "normal") or data.normal is None
        ):
            raise AttributeError(
                "PointCloudSample configured with include_normals=True but `data.normal` is missing."
            )

        pos = data.pos.float()
        normal = (
            data.normal.float() if getattr(data, "normal", None) is not None else None
        )

        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(
                f"PointCloudSample expects data.pos with shape (N, 3), got: {tuple(pos.shape)}"
            )
        if pos.shape[0] < 1:
            raise ValueError("PointCloudSample requires at least one vertex")
        if normal is not None and normal.shape != pos.shape:
            raise ValueError(
                f"PointCloudSample expects data.normal with shape {tuple(pos.shape)}, got: {tuple(normal.shape)}"
            )

        sampled_idx, valid_count = self._sample_indices(pos)
        valid_mask = torch.zeros(self.n_points, dtype=torch.bool, device=pos.device)
        valid_mask[:valid_count] = True

        pointcloud_pos = pos[sampled_idx]
        pointcloud_pos = self._normalize_points(pointcloud_pos, valid_mask)
        pointcloud_pos = self._augment_points(pointcloud_pos, valid_mask)

        data.pointcloud_pos = pointcloud_pos
        data.pointcloud_original_idx = sampled_idx.long()
        data.pointcloud_valid_mask = valid_mask
        data.pointcloud_num_points = torch.tensor(
            [self.n_points], device=pos.device, dtype=torch.long
        )

        if normal is not None:
            pointcloud_normal = normal[sampled_idx]
            normal_norm = pointcloud_normal.norm(dim=-1, keepdim=True).clamp_min(
                self.eps
            )
            pointcloud_normal = pointcloud_normal / normal_norm
            data.pointcloud_normal = pointcloud_normal

        return data


class LegacyTGCNTrainAugment(RuntimeMeshTransform):
    """Archived TCGN feature-space augmentation kept for parity diagnostics."""

    deterministic = False
    geometry_mode = "none"

    def __init__(
        self,
        x_bias: tuple[float, float] | list[float] = (-6.0, 6.0),
        y_bias: tuple[float, float] | list[float] = (-8.0, 8.0),
        z_bias: tuple[float, float] | list[float] = (-5.0, 5.0),
        degrees: float = 27.0,
    ):
        super().__init__()
        self.x_bias = tuple(float(v) for v in x_bias)
        self.y_bias = tuple(float(v) for v in y_bias)
        self.z_bias = tuple(float(v) for v in z_bias)
        if degrees < 0.0:
            raise ValueError(f"degrees must be non-negative, got {degrees}.")
        self.degrees = float(degrees)

    def apply(self, data: MeshData) -> MeshData:
        if self.stage != "train":
            return data
        data.tcgn_aug_translation = (
            float(random.uniform(*self.x_bias)),
            float(random.uniform(*self.y_bias)),
            float(random.uniform(*self.z_bias)),
        )
        data.tcgn_aug_theta = float(
            math.radians(random.uniform(-self.degrees, self.degrees))
        )
        return data


class RigidMeshTransform(RuntimeMeshTransform):
    deterministic = True
    geometry_mode = "rigid"

    def transform_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, data: MeshData) -> MeshData:
        return _apply_rigid_transform(data, matrix=self.transform_matrix())


class RotateMesh(RigidMeshTransform):
    def __init__(self, degrees: float, axis: str | Sequence[float] = "z"):
        super().__init__()
        self.degrees = float(degrees)
        self.axis = axis

    def transform_matrix(self) -> torch.Tensor:
        return _rotation_matrix(self.degrees, self.axis)


class ReflectMesh(RigidMeshTransform):
    def __init__(self, axis: str | Sequence[float] = "x"):
        super().__init__()
        self.axis = axis

    def transform_matrix(self) -> torch.Tensor:
        return _reflection_matrix(self.axis)


class RandomRotateMesh(RigidMeshTransform):
    deterministic = False

    def __init__(
        self,
        degrees: tuple[float, float] | list[float] = (-180.0, 180.0),
        seed: int | None = None,
    ):
        super().__init__()
        self.degrees = tuple(float(v) for v in degrees)
        self.seed = seed
        self.deterministic = seed is not None

    def _build_rng(self, data: MeshData) -> random.Random | None:
        if self.seed is None:
            return None
        sample_seed = int(self.seed) + _stable_hash(_sample_identifier(data))
        return random.Random(sample_seed)

    def transform_matrix(self, data: MeshData | None = None) -> torch.Tensor:
        rng = None if data is None else self._build_rng(data)
        sampled_axis = _random_unit_vector(rng=rng)
        sampled_degrees = _sample_uniform_range(self.degrees, name="degrees", rng=rng)
        return _rodrigues_rotation_matrix(axis=sampled_axis, degrees=sampled_degrees)

    def apply(self, data: MeshData) -> MeshData:
        if self.stage != "train" and self.seed is None:
            return data
        return _apply_rigid_transform(data, matrix=self.transform_matrix(data))


class RandomTCGNRotationMesh(RigidMeshTransform):
    """Apply the legacy TCGN shared-angle X/Y/Z rotation to mesh geometry."""

    deterministic = False

    def __init__(
        self,
        degrees: tuple[float, float] | list[float] | float = 27.0,
        seed: int | None = None,
    ):
        super().__init__()
        if isinstance(degrees, (int, float)):
            if degrees < 0.0:
                raise ValueError(f"degrees must be non-negative, got {degrees}.")
            self.degrees = (-float(degrees), float(degrees))
        else:
            self.degrees = tuple(float(value) for value in degrees)
            if len(self.degrees) != 2 or self.degrees[0] > self.degrees[1]:
                raise ValueError(
                    "degrees must be a scalar or a two-value range with low <= high."
                )
        self.seed = seed
        self._is_noop = self.degrees == (0.0, 0.0)
        self.deterministic = seed is not None or self._is_noop

    def _build_rng(self, data: MeshData) -> random.Random | None:
        if self.seed is None:
            return None
        sample_seed = int(self.seed) + _stable_hash(_sample_identifier(data))
        return random.Random(sample_seed)

    def transform_matrix(self, data: MeshData | None = None) -> torch.Tensor:
        if self._is_noop:
            return torch.eye(3, dtype=torch.float32)
        rng = None if data is None else self._build_rng(data)
        theta_degrees = _sample_uniform_range(
            self.degrees, name="degrees", rng=rng
        )
        rotation_x = _rotation_matrix(theta_degrees, "x")
        rotation_y = _rotation_matrix(theta_degrees, "y")
        rotation_z = _rotation_matrix(theta_degrees, "z")
        return rotation_z @ rotation_y @ rotation_x

    def apply(self, data: MeshData) -> MeshData:
        if self._is_noop:
            return data
        if self.stage != "train" and self.seed is None:
            return data
        return _apply_rigid_transform(data, matrix=self.transform_matrix(data))


class RandomRotateTranslateMesh(RuntimeMeshTransform):
    deterministic = False
    geometry_mode = "rigid"

    def __init__(
        self,
        degrees: tuple[float, float] | list[float] | float = 30.0,
        x_bias: tuple[float, float] | list[float] | float = 0.1,
        y_bias: tuple[float, float] | list[float] | float = 0.1,
        z_bias: tuple[float, float] | list[float] | float = 0.1,
        seed: int | None = None,
    ):
        super().__init__()

        def _convert_to_range(value):
            if isinstance(value, (int, float)):
                return (-float(value), float(value))
            return tuple(float(v) for v in value)

        self.degrees = _convert_to_range(degrees)
        self.x_bias = _convert_to_range(x_bias)
        self.y_bias = _convert_to_range(y_bias)
        self.z_bias = _convert_to_range(z_bias)
        self.seed = seed
        self.deterministic = seed is not None

    def _build_rng(self, data: MeshData) -> random.Random | None:
        if self.seed is None:
            return None
        sample_seed = int(self.seed) + _stable_hash(_sample_identifier(data))
        return random.Random(sample_seed)

    def apply(self, data: MeshData) -> MeshData:
        if self.stage != "train" and self.seed is None:
            return data

        rng = self._build_rng(data)

        rotation = _rodrigues_rotation_matrix(
            axis=_random_unit_vector(rng=rng),
            degrees=_sample_uniform_range(self.degrees, name="degrees", rng=rng),
        )
        translation = torch.tensor(
            [
                _sample_uniform_range(self.x_bias, name="x_bias", rng=rng),
                _sample_uniform_range(self.y_bias, name="y_bias", rng=rng),
                _sample_uniform_range(self.z_bias, name="z_bias", rng=rng),
            ],
            dtype=torch.float32,
        )
        return _apply_rigid_transform(data, matrix=rotation, translation=translation)


class NonRigidMeshTransform(RuntimeMeshTransform):
    geometry_mode = "nonrigid"

    def _validate_geometry(self, data: MeshData) -> None:
        stale_fields = _nonrigid_stale_fields(data)
        _raise_for_stale_geometry(
            data,
            transform_name=self.__class__.__name__,
            stage=self.stage,
            stale_fields=stale_fields,
        )

    def apply(self, data: MeshData) -> MeshData:
        self._validate_geometry(data)
        return self._apply_nonrigid(data)

    def _apply_nonrigid(self, data: MeshData) -> MeshData:
        raise NotImplementedError


class Vertex2EdgePromote(RuntimeMeshTransform):
    deterministic = True
    geometry_mode = "nonrigid"

    def apply(self, data: MeshData) -> MeshData:
        """Helper function to compute vertex to edge mapping for a single graph."""
        # Compute a tensor of shape (num_edges*2, 2) of values (vertex_index, edge_index)
        edge_indices = torch.arange(
            data.edge_index.shape[1], device=data.edge_index.device
        )

        # Stack vertex indices from both rows of edge_index with corresponding edge indices
        vertex2edge = torch.stack(
            [
                torch.cat([data.edge_index[0], data.edge_index[1]]),
                torch.cat([edge_indices, edge_indices]),
            ],
            dim=1,
        )

        data.vertex2edge = vertex2edge.to(data.pos.device)
        return data


class AddGaussianNoise(NonRigidMeshTransform):
    def __init__(self, std: float, mean: float = 0.0, seed: int | None = None):
        super().__init__()
        self.std = float(std)
        self.mean = float(mean)
        self.seed = seed
        self.deterministic = seed is not None

    def _apply_nonrigid(self, data: MeshData) -> MeshData:
        if self.seed is None:
            noise = torch.randn_like(data.pos)
        else:
            generator = torch.Generator(device=data.pos.device)
            sample_seed = int(self.seed) + _stable_hash(_sample_identifier(data))
            generator.manual_seed(sample_seed)
            noise = torch.randn(
                data.pos.shape,
                generator=generator,
                device=data.pos.device,
                dtype=data.pos.dtype,
            )
        data.pos = data.pos + self.mean + self.std * noise
        return data


class RandomSlideVertices(NonRigidMeshTransform):
    deterministic = False

    def __init__(
        self, rate: float = 0.2, min_fraction: float = 0.2, max_fraction: float = 0.5
    ):
        super().__init__()
        self.rate = float(rate)
        self.min_fraction = float(min_fraction)
        self.max_fraction = float(max_fraction)

    def _apply_nonrigid(self, data: MeshData) -> MeshData:
        vids = np.random.permutation(len(data.pos))
        target = int(self.rate * len(vids))
        shifted = 0
        edge_index = data.edge_index.detach().cpu().numpy().T
        for vi in vids:
            if shifted >= target:
                break
            edges = edge_index[np.where(edge_index[:, 0] == vi)]
            if len(edges) == 0:
                continue
            edge = edges[np.random.randint(len(edges))]
            vi_t = edge[1] if vi == edge[0] else edge[0]
            fraction = np.random.uniform(self.min_fraction, self.max_fraction)
            nv = data.pos[vi] + fraction * (data.pos[vi_t] - data.pos[vi])
            data.pos[vi] = nv
            shifted += 1
        return data


class GaussianKernelSmoothing(NonRigidMeshTransform):
    def __init__(
        self,
        k: int = 8,
        eta: float | None = None,
        max_samples: int = 20000,
        seed: int | None = None,
    ):
        super().__init__()
        self.k = int(k)
        self.eta = eta
        self.max_samples = int(max_samples)
        self.seed = seed
        self.deterministic = seed is not None

    def _apply_nonrigid(self, data: MeshData) -> MeshData:
        coords: torch.Tensor = data.pos.clone()
        if len(coords) > self.max_samples:
            if self.seed is None:
                raise RuntimeError(
                    "GaussianKernelSmoothing requires a seed when max_samples truncation is active."
                )
            generator = torch.Generator(device=coords.device)
            sample_seed = int(self.seed) + _stable_hash(_sample_identifier(data))
            generator.manual_seed(sample_seed)
            indices = torch.randperm(
                len(coords), generator=generator, device=coords.device
            )[: self.max_samples]
            coords = coords[indices]

        dists = torch.cdist(coords, coords, p=2)
        eta = self.eta if self.eta is not None else torch.max(dists).item()
        nearest_neighbors = torch.argsort(dists, dim=1)[:, 1 : self.k + 1]
        nearest_dists = torch.gather(dists, 1, nearest_neighbors)
        weights = torch.exp(-(nearest_dists**2) / eta)
        weights /= torch.sum(weights, dim=1, keepdim=True)
        data.pos = torch.sum(weights[:, :, None] * coords[nearest_neighbors], dim=1)
        return data
