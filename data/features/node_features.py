from data.types import MeshData
from torch_geometric.data import Batch
from typing import Literal, Union
import trimesh
import torch
from data.transforms.signature import (
    DEFAULT_SIGNATURE_APPROX,
    DEFAULT_SIGNATURE_BASIS,
    load_or_build_signature_extractor,
)

from torch_geometric.utils import scatter

from torch_cluster import knn as torch_cluster_knn
from torch_scatter import scatter_max, scatter_min


SIGNATURE_CHANNEL_INDICES = (1, 2, 3, 4, 5, 8, 10, 15)
SIGNATURE_FEATURE_DIM = len(SIGNATURE_CHANNEL_INDICES)
SIGNATURE_SAMPLE_COUNT = max(SIGNATURE_CHANNEL_INDICES) + 1


def _trimesh_from_data(data: MeshData) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=data.pos.detach().cpu(),
        faces=data.face.T.detach().cpu() if hasattr(data, "face") else None,
        process=False,
    )


def _safe_graph_scale(centered: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if centered.numel() == 0:
        return centered.new_tensor(1.0)
    bbox_extent = centered.max(dim=0).values - centered.min(dim=0).values
    bbox_diag = torch.linalg.norm(bbox_extent).clamp_min(eps)
    return bbox_diag


def _safe_radial_scale(centered: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if centered.numel() == 0:
        return centered.new_tensor(1.0)
    return torch.linalg.norm(centered, dim=-1).amax().clamp_min(eps)


def _safe_max_abs_normalize(
    values: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if values.numel() == 0:
        return values
    scale = values.abs().amax(dim=0, keepdim=True).clamp_min(eps)
    return values / scale


def _bbox_bounds(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return pos.min(dim=0).values, pos.max(dim=0).values


def _bbox_center(pos: torch.Tensor) -> torch.Tensor:
    pos_min, pos_max = _bbox_bounds(pos)
    return 0.5 * (pos_min + pos_max)


def _safe_half_extent(value_min: torch.Tensor, value_max: torch.Tensor, eps: float) -> torch.Tensor:
    return (0.5 * (value_max - value_min)).clamp_min(eps)


def compute_bbox_center_coordinates(
    data: Union[MeshData, Batch],
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian",
    eps: float = 1e-12,
):
    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(
                _compute_bbox_center_coordinates_single(graph, coordinates, eps=eps)
            )
        return torch.cat(out_list, dim=0)

    return _compute_bbox_center_coordinates_single(data, coordinates, eps=eps)


def _compute_bbox_center_coordinates_single(
    data: MeshData,
    coordinates: Literal["cartesian", "cylindrical"],
    eps: float = 1e-12,
):
    if not hasattr(data, "pos"):
        raise AttributeError("compute_bbox_center_coordinates requires `data.pos`.")

    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, 3))

    center = _bbox_center(pos)
    centered = pos - center
    scale = _safe_graph_scale(centered, eps=eps)
    centered = centered / scale

    if coordinates == "cartesian":
        return centered
    if coordinates != "cylindrical":
        raise ValueError(
            f"coordinates must be 'cartesian' or 'cylindrical', got {coordinates!r}"
        )

    radius = torch.linalg.norm(centered[:, :2], dim=-1)
    theta = torch.atan2(centered[:, 0], centered[:, 1]) / torch.pi
    z = centered[:, 2]
    return torch.stack([radius, theta, z], dim=-1)


def compute_oriented_bbox_frame(
    data: Union[MeshData, Batch],
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian",
    eps: float = 1e-12,
    angle_digits: int = 2,
):
    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(
                _compute_oriented_bbox_frame_single(
                    graph,
                    coordinates=coordinates,
                    eps=eps,
                    angle_digits=angle_digits,
                )
            )
        return torch.cat(out_list, dim=0)

    return _compute_oriented_bbox_frame_single(
        data,
        coordinates=coordinates,
        eps=eps,
        angle_digits=angle_digits,
    )


def _axis_sign_from_distribution(values: torch.Tensor, eps: float) -> float:
    skewness = torch.mean(values.pow(3))
    if torch.abs(skewness).item() <= eps:
        farthest_idx = torch.argmax(torch.abs(values))
        return 1.0 if values[farthest_idx].item() >= 0.0 else -1.0
    return 1.0 if skewness.item() >= 0.0 else -1.0


def _compute_oriented_bbox_frame_single(
    data: MeshData,
    coordinates: Literal["cartesian", "cylindrical"],
    eps: float = 1e-12,
    angle_digits: int = 2,
):
    if not hasattr(data, "pos"):
        raise AttributeError("compute_oriented_bbox_frame requires `data.pos`.")

    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, 3))

    if coordinates not in ("cartesian", "cylindrical"):
        raise ValueError(
            f"coordinates must be 'cartesian' or 'cylindrical', got {coordinates!r}"
        )

    try:
        mesh = _trimesh_from_data(data)
        to_origin, extents = trimesh.bounds.oriented_bounds(
            mesh,
            angle_digits=angle_digits,
            ordered=True,
        )
        aligned_np = trimesh.transform_points(
            pos.detach().cpu().numpy(),
            to_origin,
        )
        aligned = torch.as_tensor(aligned_np, dtype=pos.dtype, device=pos.device)
        extents_t = torch.as_tensor(extents, dtype=pos.dtype, device=pos.device)
    except Exception:
        return _compute_canonical_frame_single(data, coordinates=coordinates, eps=eps)

    sign = aligned.new_ones(3)
    for axis_idx in range(3):
        sign[axis_idx] = _axis_sign_from_distribution(aligned[:, axis_idx], eps=eps)
    aligned = aligned * sign
    if sign.prod().item() < 0.0:
        aligned[:, 2] = -aligned[:, 2]

    scale = torch.linalg.norm(extents_t).clamp_min(eps)
    aligned = aligned / scale

    if coordinates == "cartesian":
        return aligned

    radius = torch.linalg.norm(aligned[:, :2], dim=-1)
    theta = torch.atan2(aligned[:, 0], aligned[:, 1]) / torch.pi
    z = aligned[:, 2]
    return torch.stack([radius, theta, z], dim=-1)


def compute_com_fps_anchor_distances(
    data: Union[MeshData, Batch],
    num_fps_anchors: int = 3,
    eps: float = 1e-12,
):
    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(
                _compute_com_fps_anchor_distances_single(
                    graph,
                    num_fps_anchors=num_fps_anchors,
                    eps=eps,
                )
            )
        return torch.cat(out_list, dim=0)

    return _compute_com_fps_anchor_distances_single(
        data,
        num_fps_anchors=num_fps_anchors,
        eps=eps,
    )


def _greedy_fps_anchor_indices(
    pos: torch.Tensor,
    num_anchors: int,
    seed_index: int,
) -> torch.Tensor:
    if pos.shape[0] == 0 or num_anchors <= 0:
        return pos.new_zeros((0,), dtype=torch.long)

    selected = [int(seed_index)]
    min_dist = torch.cdist(pos[seed_index].unsqueeze(0), pos, p=2).squeeze(0)
    while len(selected) < min(num_anchors, pos.shape[0]):
        next_index = int(torch.argmax(min_dist).item())
        if next_index in selected:
            break
        selected.append(next_index)
        next_dist = torch.cdist(pos[next_index].unsqueeze(0), pos, p=2).squeeze(0)
        min_dist = torch.minimum(min_dist, next_dist)
    return torch.tensor(selected, device=pos.device, dtype=torch.long)


def _compute_com_fps_anchor_distances_single(
    data: MeshData,
    num_fps_anchors: int = 3,
    eps: float = 1e-12,
):
    if not hasattr(data, "pos"):
        raise AttributeError("compute_com_fps_anchor_distances requires `data.pos`.")

    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, num_fps_anchors + 1))

    center = pos.mean(dim=0, keepdim=True)
    dist_to_center = torch.linalg.norm(pos - center, dim=-1)
    seed_index = int(torch.argmax(dist_to_center).item())
    fps_indices = _greedy_fps_anchor_indices(pos, num_fps_anchors, seed_index)
    anchors = torch.cat([center, pos[fps_indices]], dim=0)

    distances = torch.cdist(pos, anchors, p=2)
    scale = _safe_radial_scale(pos - center, eps=eps)
    return distances / scale


def compute_distance_to_bbox_mid_sagittal_plane(
    data: Union[MeshData, Batch],
    eps: float = 1e-12,
):
    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(
                _compute_distance_to_bbox_mid_sagittal_plane_single(graph, eps=eps)
            )
        return torch.cat(out_list, dim=0)

    return _compute_distance_to_bbox_mid_sagittal_plane_single(data, eps=eps)


def _compute_distance_to_bbox_mid_sagittal_plane_single(
    data: MeshData,
    eps: float = 1e-12,
):
    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, 1))

    pos_min, pos_max = _bbox_bounds(pos)
    x_center = 0.5 * (pos_min[0] + pos_max[0])
    half_extent_x = _safe_half_extent(pos_min[0], pos_max[0], eps)
    distance = torch.abs(pos[:, 0] - x_center) / half_extent_x
    return distance.unsqueeze(-1).clamp_max(1.0)


def compute_signed_bbox_midline_offset(
    data: Union[MeshData, Batch],
    eps: float = 1e-12,
):
    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(_compute_signed_bbox_midline_offset_single(graph, eps=eps))
        return torch.cat(out_list, dim=0)

    return _compute_signed_bbox_midline_offset_single(data, eps=eps)


def _compute_signed_bbox_midline_offset_single(
    data: MeshData,
    eps: float = 1e-12,
):
    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, 1))

    pos_min, pos_max = _bbox_bounds(pos)
    x_center = 0.5 * (pos_min[0] + pos_max[0])
    half_extent_x = _safe_half_extent(pos_min[0], pos_max[0], eps)
    offset = (pos[:, 0] - x_center) / half_extent_x
    return offset.unsqueeze(-1).clamp(min=-1.0, max=1.0)


def compute_bbox_vertical_position(
    data: Union[MeshData, Batch],
    eps: float = 1e-12,
):
    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(_compute_bbox_vertical_position_single(graph, eps=eps))
        return torch.cat(out_list, dim=0)

    return _compute_bbox_vertical_position_single(data, eps=eps)


def _compute_bbox_vertical_position_single(
    data: MeshData,
    eps: float = 1e-12,
):
    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, 1))

    z = pos[:, 2]
    z_min = z.min()
    z_max = z.max()
    denom = (z_max - z_min).clamp_min(eps)
    vertical = (z - z_min) / denom
    return vertical.unsqueeze(-1)


def compute_canonical_frame(
    data: Union[MeshData, Batch],
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian",
    eps: float = 1e-12,
):
    """Compute a deterministic PCA-based canonical frame per mesh.

    The frame is centered at the mesh centroid. Principal axes come from PCA and
    their signs are resolved using projection skewness so the frame is stable
    across runs while still permitting reflection-symmetry breaking.
    """

    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(_compute_canonical_frame_single(graph, coordinates, eps=eps))
        return torch.cat(out_list, dim=0)

    return _compute_canonical_frame_single(data, coordinates, eps=eps)


def _orient_axis_by_skewness(
    centered: torch.Tensor,
    axis: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Orient a PCA axis by the skewness of the point distribution along that axis."""
    axis = torch.nn.functional.normalize(axis, p=2, dim=0, eps=eps)
    projection = centered.matmul(axis)
    skewness = torch.mean(projection.pow(3))
    if torch.abs(skewness).item() <= eps:
        farthest_idx = torch.argmax(torch.abs(projection))
        if projection[farthest_idx].item() < 0.0:
            axis = -axis
        return axis
    return axis if skewness.item() >= 0.0 else -axis


def _get_face_area_or_compute(
    data: MeshData,
    pos: torch.Tensor,
    face: torch.Tensor,
) -> torch.Tensor:
    if hasattr(data, "face_area") and data.face_area is not None:
        face_area = data.face_area.float().to(device=pos.device)
        if face_area.numel() == face.shape[1]:
            return face_area
    return _compute_face_area_from_geometry(pos, face)


def _compute_vertex_area_weights(
    data: MeshData,
    pos: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    num_nodes = int(pos.shape[0])
    if num_nodes == 0:
        return pos.new_zeros((0,))

    if not hasattr(data, "face") or data.face is None or data.face.numel() == 0:
        return pos.new_full((num_nodes,), 1.0 / float(num_nodes))

    face = data.face.long()
    face_area = _get_face_area_or_compute(data, pos, face)
    vertex_weights = pos.new_zeros((num_nodes,))
    face_area_thirds = (face_area / 3.0).to(device=pos.device)
    for vertex_idx in range(3):
        vertex_weights = vertex_weights + scatter(
            face_area_thirds,
            face[vertex_idx],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

    total_weight = vertex_weights.sum().clamp_min(eps)
    return vertex_weights / total_weight


def _compute_weighted_centroid(
    pos: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return (weights.unsqueeze(-1) * pos).sum(dim=0) / weights.sum().clamp_min(eps)


def _compute_weighted_covariance(
    centered: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    weighted_centered = centered * weights.unsqueeze(-1)
    return centered.t().matmul(weighted_centered) / weights.sum().clamp_min(eps)


def _relative_eigenvalue_gap(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float,
) -> float:
    denom = torch.maximum(first.abs(), second.abs()).clamp_min(eps)
    return torch.abs(first - second).div(denom).item()


def _axis_asymmetry_score(
    centered: torch.Tensor,
    axis: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    axis = torch.nn.functional.normalize(axis, p=2, dim=0, eps=eps)
    projection = centered.matmul(axis)
    weighted_third = torch.abs(torch.sum(weights * projection.pow(3)))
    extent_asymmetry = torch.abs(projection.max() - (-projection).max())
    return weighted_third + 0.1 * extent_asymmetry


def _orient_axis_by_weighted_asymmetry(
    centered: torch.Tensor,
    axis: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    axis = torch.nn.functional.normalize(axis, p=2, dim=0, eps=eps)
    projection = centered.matmul(axis)
    weighted_skewness = torch.sum(weights * projection.pow(3))
    if torch.abs(weighted_skewness).item() > eps:
        return axis if weighted_skewness.item() >= 0.0 else -axis

    support_idx = torch.argmax(weights * projection.abs())
    if projection[support_idx].item() < 0.0:
        axis = -axis
    return axis


def _fallback_orthogonal_axis(axis: torch.Tensor, eps: float) -> torch.Tensor:
    basis = torch.eye(3, device=axis.device, dtype=axis.dtype)
    anchor = basis[torch.argmin(torch.abs(axis))]
    orthogonal = anchor - torch.dot(anchor, axis) * axis
    return torch.nn.functional.normalize(orthogonal, p=2, dim=0, eps=eps)


def _build_plane_basis(
    axes: torch.Tensor,
    primary_axis: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    basis_vectors = []
    for axis in axes:
        candidate = axis - torch.dot(axis, primary_axis) * primary_axis
        candidate_norm = torch.linalg.norm(candidate)
        if candidate_norm.item() <= eps:
            continue
        candidate = candidate / candidate_norm
        if basis_vectors and torch.abs(torch.dot(candidate, basis_vectors[0])).item() > 1.0 - 1e-4:
            continue
        basis_vectors.append(candidate)
        if len(basis_vectors) == 2:
            break

    if not basis_vectors:
        basis_vectors.append(_fallback_orthogonal_axis(primary_axis, eps=eps))
    if len(basis_vectors) == 1:
        secondary = torch.cross(primary_axis, basis_vectors[0], dim=0)
        secondary = torch.nn.functional.normalize(secondary, p=2, dim=0, eps=eps)
        basis_vectors.append(secondary)

    return torch.stack(basis_vectors, dim=0)


def _resolve_axis_from_plane(
    centered: torch.Tensor,
    plane_basis: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    angle_steps: int = 181,
) -> torch.Tensor:
    if plane_basis.shape[0] == 1:
        return _orient_axis_by_weighted_asymmetry(
            centered,
            plane_basis[0],
            weights,
            eps=eps,
        )

    angles = torch.linspace(
        0.0,
        torch.pi,
        steps=angle_steps,
        device=centered.device,
        dtype=centered.dtype,
    )
    candidates = (
        torch.cos(angles).unsqueeze(-1) * plane_basis[0].unsqueeze(0)
        + torch.sin(angles).unsqueeze(-1) * plane_basis[1].unsqueeze(0)
    )
    projections = centered.matmul(candidates.t())
    weighted_third = torch.abs((weights.unsqueeze(-1) * projections.pow(3)).sum(dim=0))
    extent_asymmetry = torch.abs(projections.max(dim=0).values - (-projections).max(dim=0).values)
    scores = weighted_third + 0.1 * extent_asymmetry
    best_axis = candidates[torch.argmax(scores)]
    return _orient_axis_by_weighted_asymmetry(centered, best_axis, weights, eps=eps)


def _compute_canonical_frame_single(
    data: MeshData,
    coordinates: Literal["cartesian", "cylindrical"],
    eps: float = 1e-12,
):
    if not hasattr(data, "pos"):
        raise AttributeError("compute_canonical_frame requires `data.pos`.")

    pos = data.pos
    if pos.numel() == 0:
        return pos.new_zeros((0, 3))

    if coordinates not in ("cartesian", "cylindrical"):
        raise ValueError(
            f"coordinates must be 'cartesian' or 'cylindrical', got {coordinates!r}"
        )

    centroid = pos.mean(dim=0)
    centered = pos - centroid
    scale = _safe_graph_scale(centered, eps=eps)

    if centered.size(0) < 3:
        rotated = centered
    else:
        cov = centered.t().matmul(centered) / float(centered.size(0))
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        axes = evecs[:, order].t()

        x_axis = _orient_axis_by_skewness(centered, axes[0], eps=eps)
        y_axis = _orient_axis_by_skewness(centered, axes[1], eps=eps)

        y_axis = y_axis - torch.dot(y_axis, x_axis) * x_axis
        y_axis_norm = torch.linalg.norm(y_axis)
        if y_axis_norm.item() <= eps:
            y_axis = axes[1]
        y_axis = torch.nn.functional.normalize(y_axis, p=2, dim=0, eps=eps)

        z_axis = torch.cross(x_axis, y_axis, dim=0)
        z_axis = torch.nn.functional.normalize(z_axis, p=2, dim=0, eps=eps)
        if torch.dot(z_axis, axes[2]).item() < 0.0:
            y_axis = -y_axis
            z_axis = -z_axis

        rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)
        rotated = centered.matmul(rotation)

    rotated = rotated / scale

    if coordinates == "cartesian":
        return rotated

    radius = torch.linalg.norm(rotated[:, :2], dim=-1)
    theta = torch.atan2(rotated[:, 0], rotated[:, 1]) / torch.pi
    z = rotated[:, 2]
    return torch.stack([radius, theta, z], dim=-1)


def compute_stable_canonical_frame(
    data: Union[MeshData, Batch],
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian",
    eps: float = 1e-12,
    ambiguity_tol: float = 5e-2,
):
    """Compute a weighted canonical frame with deterministic ambiguity handling."""

    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(
                _compute_stable_canonical_frame_single(
                    graph,
                    coordinates=coordinates,
                    eps=eps,
                    ambiguity_tol=ambiguity_tol,
                )
            )
        return torch.cat(out_list, dim=0)

    return _compute_stable_canonical_frame_single(
        data,
        coordinates=coordinates,
        eps=eps,
        ambiguity_tol=ambiguity_tol,
    )


def _compute_stable_canonical_frame_single(
    data: MeshData,
    coordinates: Literal["cartesian", "cylindrical"],
    eps: float = 1e-12,
    ambiguity_tol: float = 1e-2,
):
    if not hasattr(data, "pos"):
        raise AttributeError("compute_stable_canonical_frame requires `data.pos`.")

    pos = data.pos.float()
    if pos.numel() == 0:
        return pos.new_zeros((0, 3))

    if coordinates not in ("cartesian", "cylindrical"):
        raise ValueError(
            f"coordinates must be 'cartesian' or 'cylindrical', got {coordinates!r}"
        )

    weights = _compute_vertex_area_weights(data, pos, eps=eps)
    centroid = _compute_weighted_centroid(pos, weights, eps=eps)
    centered = pos - centroid

    if centered.size(0) < 3:
        rotated = centered
    else:
        cov = _compute_weighted_covariance(centered, weights, eps=eps)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        axes = evecs[:, order].t()

        top_gap = _relative_eigenvalue_gap(evals[0], evals[1], eps=eps)
        if top_gap <= ambiguity_tol:
            x_axis = _resolve_axis_from_plane(centered, axes[:2], weights, eps=eps)
        else:
            x_axis = _orient_axis_by_weighted_asymmetry(
                centered,
                axes[0],
                weights,
                eps=eps,
            )

        second_gap = _relative_eigenvalue_gap(evals[1], evals[2], eps=eps)
        if top_gap <= ambiguity_tol or second_gap <= ambiguity_tol:
            plane_basis = _build_plane_basis(axes, x_axis, eps=eps)
            y_axis = _resolve_axis_from_plane(centered, plane_basis, weights, eps=eps)
        else:
            y_axis = _orient_axis_by_weighted_asymmetry(
                centered,
                axes[1],
                weights,
                eps=eps,
            )

        y_axis = y_axis - torch.dot(y_axis, x_axis) * x_axis
        y_axis_norm = torch.linalg.norm(y_axis)
        if y_axis_norm.item() <= eps:
            plane_basis = _build_plane_basis(axes, x_axis, eps=eps)
            y_axis = plane_basis[0]
        y_axis = torch.nn.functional.normalize(y_axis, p=2, dim=0, eps=eps)

        z_axis = torch.cross(x_axis, y_axis, dim=0)
        z_axis = torch.nn.functional.normalize(z_axis, p=2, dim=0, eps=eps)

        rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)
        rotated = centered.matmul(rotation)

    scale = _safe_graph_scale(rotated, eps=eps)
    rotated = rotated / scale

    if coordinates == "cartesian":
        return rotated

    radius = torch.linalg.norm(rotated[:, :2], dim=-1)
    theta = torch.atan2(rotated[:, 0], rotated[:, 1]) / torch.pi
    z = rotated[:, 2]
    return torch.stack([radius, theta, z], dim=-1)


def compute_stable_pca_frame_diagnostics(
    data: MeshData,
    eps: float = 1e-12,
    ambiguity_tol: float = 5e-2,
) -> dict[str, torch.Tensor]:
    """Return the frame and ambiguity diagnostics used by the stable PCA features.

    ``frame`` stores the resolved principal axes as columns in world coordinates.
    Thus, for a rigid rotation ``R``, an unambiguous frame should satisfy
    ``frame(RX) == R @ frame(X)``.  The eigengaps and sign margins are
    dimensionless, so they can be compared across meshes.
    """
    if not hasattr(data, "pos"):
        raise AttributeError("compute_stable_pca_frame_diagnostics requires `data.pos`.")

    pos = data.pos.float()
    if pos.numel() == 0:
        zeros = pos.new_zeros(3)
        return {
            "frame": pos.new_zeros((3, 3)),
            "eigenvalues": zeros,
            "relative_eigengaps": pos.new_zeros(2),
            "sign_margins": pos.new_zeros(2),
            "centroid": zeros,
        }

    weights = _compute_vertex_area_weights(data, pos, eps=eps)
    centroid = _compute_weighted_centroid(pos, weights, eps=eps)
    centered = pos - centroid

    if centered.size(0) < 3:
        return {
            "frame": torch.eye(3, dtype=pos.dtype, device=pos.device),
            "eigenvalues": pos.new_zeros(3),
            "relative_eigengaps": pos.new_zeros(2),
            "sign_margins": pos.new_zeros(2),
            "centroid": centroid,
        }

    cov = _compute_weighted_covariance(centered, weights, eps=eps)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    axes = evecs[:, order].t()

    top_gap = _relative_eigenvalue_gap(evals[0], evals[1], eps=eps)
    if top_gap <= ambiguity_tol:
        x_axis = _resolve_axis_from_plane(centered, axes[:2], weights, eps=eps)
    else:
        x_axis = _orient_axis_by_weighted_asymmetry(
            centered, axes[0], weights, eps=eps
        )

    second_gap = _relative_eigenvalue_gap(evals[1], evals[2], eps=eps)
    if top_gap <= ambiguity_tol or second_gap <= ambiguity_tol:
        plane_basis = _build_plane_basis(axes, x_axis, eps=eps)
        y_axis = _resolve_axis_from_plane(centered, plane_basis, weights, eps=eps)
    else:
        y_axis = _orient_axis_by_weighted_asymmetry(
            centered, axes[1], weights, eps=eps
        )

    y_axis = y_axis - torch.dot(y_axis, x_axis) * x_axis
    if torch.linalg.norm(y_axis).item() <= eps:
        y_axis = _build_plane_basis(axes, x_axis, eps=eps)[0]
    y_axis = torch.nn.functional.normalize(y_axis, p=2, dim=0, eps=eps)
    z_axis = torch.nn.functional.normalize(
        torch.cross(x_axis, y_axis, dim=0), p=2, dim=0, eps=eps
    )
    frame = torch.stack([x_axis, y_axis, z_axis], dim=1)

    # The signed third moments are exactly the quantities used by the normal
    # sign resolver.  Divide by the corresponding PCA scale to make them
    # comparable between meshes.  Only the independently resolved x/y signs
    # are reported; z follows from their right-handed cross product.
    projections = centered.matmul(frame[:, :2])
    weighted_third = (weights.unsqueeze(-1) * projections.pow(3)).sum(dim=0)
    sign_scales = evals[:2].clamp_min(eps).pow(1.5)
    sign_margins = weighted_third.abs() / sign_scales

    return {
        "frame": frame,
        "eigenvalues": evals,
        "relative_eigengaps": pos.new_tensor([top_gap, second_gap]),
        "sign_margins": sign_margins,
        "centroid": centroid,
    }


def compute_dental_frame(
    data: Union[MeshData, Batch],
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian",
    eps: float = 1e-12,
):
    """Compute per-node coordinates in an anatomical (PCA-based) dental frame.

    This reproduces the logic from `visualize.ipynb`:
      1) Center vertices.
      2) PCA: component0 -> left-right (X), component1 -> anterior-posterior (Y),
         component2 -> plane normal (Z).
      3) Resolve sign ambiguity of Y via a quadratic fit of y_local vs x_local.
      4) Rotate into the new frame and shift so that the most posterior point has y=0.

    Args:
      data: PyG `MeshData` or `Batch` with `data.pos` (N,3).
      coordinates: Either:
        - "cartesian": returns (x, y, z) in the new frame.
        - "cylindrical": returns (r, theta, z) where
            z = new_verts[:, 2]
            theta = abs(atan2(new_verts[:, 0], new_verts[:, 1]))
            r = sqrt(new_verts[:, 0]^2 + new_verts[:, 1]^2)
      eps: Numerical epsilon for normalisation.

    Returns:
      Tensor of shape (num_nodes, 3) in the requested coordinate system.
    """

    if isinstance(data, Batch):
        out_list = []
        for graph in data.to_data_list():
            out_list.append(_compute_dental_frame_single(graph, coordinates, eps=eps))
        return torch.cat(out_list, dim=0)

    return _compute_dental_frame_single(data, coordinates, eps=eps)


def _compute_dental_frame_single(
    data: MeshData,
    coordinates: Literal["cartesian", "cylindrical"],
    eps: float = 1e-12,
):
    if not hasattr(data, "pos"):
        raise AttributeError("compute_dental_frame requires `data.pos`.")

    pos = data.pos
    if pos.numel() == 0:
        return pos.new_zeros((0, 3))

    if coordinates not in ("cartesian", "cylindrical"):
        raise ValueError(
            f"coordinates must be 'cartesian' or 'cylindrical', got {coordinates!r}"
        )

    centroid = pos.mean(dim=0)
    centered = pos - centroid

    # Degenerate cases: too few points to define a stable PCA frame.
    if centered.size(0) < 3:
        rotated = centered
        min_y = rotated[:, 1].min()
        new_verts = rotated + rotated.new_tensor([0.0, -min_y, 0.0])
        if coordinates == "cartesian":
            return new_verts
        r = torch.linalg.norm(new_verts[:, :2], dim=-1)
        theta = torch.abs(torch.atan2(new_verts[:, 0], new_verts[:, 1]))
        z = new_verts[:, 2]
        return torch.stack([r, theta, z], dim=-1)

    # PCA via covariance eigen-decomposition (pure torch).
    # cov = (X^T X) / N (symmetric 3x3)
    cov = centered.t().matmul(centered) / float(centered.size(0))
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    order = torch.argsort(evals, descending=True)
    axes = evecs[:, order].t()  # (3,3) rows are principal axes

    x_axis = axes[0]
    y_axis = axes[1]
    z_axis = axes[2]

    def _orthonormalise(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        z = torch.nn.functional.normalize(z, p=2, dim=0, eps=eps)
        y = torch.nn.functional.normalize(y, p=2, dim=0, eps=eps)
        x = torch.cross(y, z, dim=0)
        x = torch.nn.functional.normalize(x, p=2, dim=0, eps=eps)
        z = torch.cross(x, y, dim=0)
        z = torch.nn.functional.normalize(z, p=2, dim=0, eps=eps)
        return x, y, z

    x_axis, y_axis, z_axis = _orthonormalise(x_axis, y_axis, z_axis)

    # Resolve sign ambiguity for Y using a quadratic fit y = a x^2 + b x + c.
    # If concavity a > 0, we flip Y to make +Y point anterior.
    # (Matches logic in visualize.ipynb.)
    R_rows = torch.stack([x_axis, y_axis, z_axis], dim=0)  # rows
    coords = centered.matmul(R_rows.t())
    x_local = coords[:, 0]
    y_local = coords[:, 1]
    try:
        A = torch.stack([x_local * x_local, x_local, torch.ones_like(x_local)], dim=1)
        sol = torch.linalg.lstsq(A, y_local.unsqueeze(1)).solution.squeeze(1)
        concavity = sol[0]
        if concavity.item() > 0:
            y_axis = -y_axis
            x_axis, y_axis, z_axis = _orthonormalise(x_axis, y_axis, z_axis)
    except Exception:
        pass

    # Columns are the new axes [X, Y, Z]
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (3,3)
    rotated = centered.matmul(R)

    # Shift so the most posterior point has y = 0.
    min_y = rotated[:, 1].min()
    new_verts = rotated + rotated.new_tensor([0.0, -min_y, 0.0])

    if coordinates == "cartesian":
        return new_verts

    r = torch.linalg.norm(new_verts[:, :2], dim=-1)
    theta = torch.atan2(new_verts[:, 0], new_verts[:, 1])
    z = new_verts[:, 2]
    return torch.stack([r, theta, z], dim=-1)


def compute_dental_pca_frame_diagnostics(
    data: MeshData,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return the PCA frame and sign diagnostic used by ``dental_frame_*``.

    The axes and concavity-based Y-sign resolution intentionally duplicate
    ``_compute_dental_frame_single`` so a stability evaluation measures the
    exact coordinate convention consumed by Teeth3DS models. ``frame`` stores
    axes as columns in world coordinates.
    """
    if not hasattr(data, "pos"):
        raise AttributeError("compute_dental_pca_frame_diagnostics requires `data.pos`.")

    pos = data.pos
    if pos.numel() == 0:
        zeros = pos.new_zeros(3)
        return {
            "frame": pos.new_zeros((3, 3)),
            "eigenvalues": zeros,
            "relative_eigengaps": pos.new_zeros(2),
            "sign_margins": pos.new_zeros(1),
            "centroid": zeros,
        }

    centroid = pos.mean(dim=0)
    centered = pos - centroid
    if centered.size(0) < 3:
        return {
            "frame": torch.eye(3, dtype=pos.dtype, device=pos.device),
            "eigenvalues": pos.new_zeros(3),
            "relative_eigengaps": pos.new_zeros(2),
            "sign_margins": pos.new_zeros(1),
            "centroid": centroid,
        }

    cov = centered.t().matmul(centered) / float(centered.size(0))
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    axes = evecs[:, order].t()
    x_axis, y_axis, z_axis = axes[0], axes[1], axes[2]

    def _orthonormalise(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        z = torch.nn.functional.normalize(z, p=2, dim=0, eps=eps)
        y = torch.nn.functional.normalize(y, p=2, dim=0, eps=eps)
        x = torch.cross(y, z, dim=0)
        x = torch.nn.functional.normalize(x, p=2, dim=0, eps=eps)
        z = torch.cross(x, y, dim=0)
        z = torch.nn.functional.normalize(z, p=2, dim=0, eps=eps)
        return x, y, z

    x_axis, y_axis, z_axis = _orthonormalise(x_axis, y_axis, z_axis)
    coords = centered.matmul(torch.stack([x_axis, y_axis, z_axis], dim=0).t())
    x_local, y_local = coords[:, 0], coords[:, 1]
    concavity = pos.new_zeros(())
    try:
        design = torch.stack(
            [x_local * x_local, x_local, torch.ones_like(x_local)], dim=1
        )
        concavity = torch.linalg.lstsq(design, y_local.unsqueeze(1)).solution.squeeze(1)[0]
        if concavity.item() > 0:
            y_axis = -y_axis
            x_axis, y_axis, z_axis = _orthonormalise(x_axis, y_axis, z_axis)
    except Exception:
        pass

    # The quadratic coefficient has inverse-length units. Scaling it by the
    # PCA variances gives a dimensionless margin for the dental Y-sign choice.
    sign_margin = concavity.abs() * evals[0] / evals[1].clamp_min(eps).sqrt()
    return {
        "frame": torch.stack([x_axis, y_axis, z_axis], dim=1),
        "eigenvalues": evals,
        "relative_eigengaps": pos.new_tensor(
            [
                _relative_eigenvalue_gap(evals[0], evals[1], eps=eps),
                _relative_eigenvalue_gap(evals[1], evals[2], eps=eps),
            ]
        ),
        "sign_margins": sign_margin.unsqueeze(0),
        "centroid": centroid,
    }


def compute_hks(data: Union[MeshData, Batch]):
    return _compute_signature_feature(data, kernel="heat")


def _compute_hks_single(data: MeshData):
    return _compute_signature_feature_single(data, kernel="heat")


def compute_wave_signature(data: Union[MeshData, Batch]):
    return _compute_signature_feature(data, kernel="wave")


def compute_spectral_coordinates(data: Union[MeshData, Batch]):
    if isinstance(data, Batch):
        coords_list = []
        for graph in data.to_data_list():
            coords_list.append(_compute_spectral_coordinates_single(graph))
        return torch.cat(coords_list, dim=0)

    return _compute_spectral_coordinates_single(data)


def _compute_signature_feature(
    data: Union[MeshData, Batch],
    *,
    kernel: Literal["heat", "wave"],
):
    if isinstance(data, Batch):
        feature_list = []
        for graph in data.to_data_list():
            feature_list.append(_compute_signature_feature_single(graph, kernel=kernel))
        return torch.cat(feature_list, dim=0)

    return _compute_signature_feature_single(data, kernel=kernel)


def _compute_signature_feature_single(
    data: MeshData,
    *,
    kernel: Literal["heat", "wave"],
) -> torch.Tensor:
    extractor = _load_signature_extractor(data)
    signature = torch.tensor(
        extractor.signatures(SIGNATURE_SAMPLE_COUNT, kernel, return_x_ticks=False),
        dtype=data.pos.dtype,
        device=data.pos.device,
    )
    return _select_signature_channels(signature)


def _compute_spectral_coordinates_single(data: MeshData) -> torch.Tensor:
    extractor = _load_signature_extractor(data)
    evecs = torch.tensor(
        extractor.evecs,
        dtype=data.pos.dtype,
        device=data.pos.device,
    )
    nontrivial = evecs[:, 1 : 1 + SIGNATURE_FEATURE_DIM]
    # nontrivial = evecs[:, :SIGNATURE_FEATURE_DIM]
    if nontrivial.shape[1] < SIGNATURE_FEATURE_DIM:
        padding = nontrivial.new_zeros(
            (nontrivial.shape[0], SIGNATURE_FEATURE_DIM - nontrivial.shape[1])
        )
        nontrivial = torch.cat([nontrivial, padding], dim=1)
    return _stabilize_eigenvector_signs(nontrivial)


def _load_signature_extractor(data: MeshData):
    approx = getattr(data, "signature_approx", DEFAULT_SIGNATURE_APPROX)
    n_basis = int(getattr(data, "signature_basis_count", DEFAULT_SIGNATURE_BASIS))
    return load_or_build_signature_extractor(
        data,
        approx=approx,
        n_basis=n_basis,
        cache_if_possible=True,  # Enable caching to avoid recomputation
        uncached_warning=True,  # Warn if we have to compute a signature extractor from scratch, since this can be expensive
    )


def _select_signature_channels(signature: torch.Tensor) -> torch.Tensor:
    selected = signature[:, list(SIGNATURE_CHANNEL_INDICES)]
    col_min = selected.min(dim=0, keepdim=True).values
    col_max = selected.max(dim=0, keepdim=True).values
    denom = (col_max - col_min).clamp_min(1e-12)
    return (selected - col_min) / denom


def _stabilize_eigenvector_signs(evecs: torch.Tensor) -> torch.Tensor:
    if evecs.numel() == 0:
        return evecs

    max_indices = evecs.abs().argmax(dim=0)
    column_indices = torch.arange(evecs.shape[1], device=evecs.device)
    signs = torch.sign(evecs[max_indices, column_indices]).unsqueeze(0)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return evecs * signs


def compute_pointwise_area(data: Union[MeshData, Batch]):
    # Handle batch processing
    if isinstance(data, Batch):
        area_points = []
        vertex2faces = []
        graphs = data.to_data_list()

        vertex_offset = 0
        face_offset = 0

        for graph in graphs:
            area_point, vertex2face = _compute_pointwise_area_single(graph)

            # Adjust indices for batch concatenation
            adjusted_vertex2face = vertex2face.clone()
            adjusted_vertex2face[:, 0] += vertex_offset
            adjusted_vertex2face[:, 1] += face_offset

            area_points.append(area_point)
            vertex2faces.append(adjusted_vertex2face)

            vertex_offset += graph.pos.shape[0]
            face_offset += graph.face.shape[1]

        return torch.cat(area_points, dim=0), torch.cat(vertex2faces, dim=0)

    # Handle single graph
    return _compute_pointwise_area_single(data)


def _compute_pointwise_area_single(data: MeshData):
    """Helper function to compute pointwise area for a single graph."""
    mesh = _trimesh_from_data(data)

    vertex = (
        torch.arange(mesh.vertex_faces.shape[0], device=data.pos.device)
        .unsqueeze(-1)
        .expand(-1, mesh.vertex_faces.shape[1])
        .flatten()
        .unsqueeze(-1)
    )
    face = torch.tensor(
        mesh.vertex_faces.reshape(-1), device=data.pos.device
    ).unsqueeze(-1)
    vertex2face = torch.cat([vertex, face], dim=-1)
    vertex2face = vertex2face[vertex2face[:, 1] != -1]

    area_point = scatter(
        data.face_area[vertex2face[:, 1]],
        vertex2face[:, 0],
        dim=0,
        dim_size=mesh.vertices.shape[0],
        reduce="mean",
    ).to(data.pos.device)

    vertex2face = vertex2face.to(data.pos.device)

    return area_point, vertex2face


def compute_background_binary(data: Union[MeshData, Batch]):
    # Vertex labels == 0 are background
    if not hasattr(data, "vertex_labels"):
        raise AttributeError("compute_background_binary requires `data.vertex_labels`.")
    return (data.vertex_labels == 0).float()


def compute_gaussian_curvature(data: Union[MeshData, Batch], eps: float = 1e-12):
    if isinstance(data, Batch):
        curvatures = []
        for graph in data.to_data_list():
            curvatures.append(_compute_curvature_invariants_single(graph, eps=eps)[0])
        return torch.cat(curvatures, dim=0)

    return _compute_curvature_invariants_single(data, eps=eps)[0]


def compute_principal_curvatures(data: Union[MeshData, Batch], eps: float = 1e-12):
    if isinstance(data, Batch):
        curvatures = []
        for graph in data.to_data_list():
            curvatures.append(_compute_curvature_invariants_single(graph, eps=eps)[1])
        return torch.cat(curvatures, dim=0)

    return _compute_curvature_invariants_single(data, eps=eps)[1]


def _compute_face_area_from_geometry(pos: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    v0 = pos[face[0].long()]
    v1 = pos[face[1].long()]
    v2 = pos[face[2].long()]
    return 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)


def _compute_curvature_invariants_single(data: MeshData, eps: float = 1e-12):
    if not hasattr(data, "pos") or data.pos is None:
        raise AttributeError("Curvature features require `data.pos`.")
    if not hasattr(data, "face") or data.face is None:
        raise AttributeError("Curvature features require `data.face`.")

    pos = data.pos.float()
    face = data.face.long()
    num_nodes = int(pos.shape[0])

    if num_nodes == 0:
        empty_scalar = pos.new_zeros((0, 1))
        empty_pair = pos.new_zeros((0, 2))
        return empty_scalar, empty_pair

    if face.numel() == 0:
        zero_scalar = pos.new_zeros((num_nodes, 1))
        zero_pair = pos.new_zeros((num_nodes, 2))
        return zero_scalar, zero_pair

    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    e01 = v1 - v0
    e02 = v2 - v0
    e10 = v0 - v1
    e12 = v2 - v1
    e20 = v0 - v2
    e21 = v1 - v2

    cross_0 = torch.cross(e01, e02, dim=-1)
    double_area = torch.linalg.norm(cross_0, dim=-1).clamp_min(eps)

    cot0 = (e01 * e02).sum(dim=-1) / double_area
    cot1 = (e12 * e10).sum(dim=-1) / double_area
    cot2 = (e20 * e21).sum(dim=-1) / double_area

    angle0 = torch.atan2(double_area, (e01 * e02).sum(dim=-1))
    angle1 = torch.atan2(double_area, (e12 * e10).sum(dim=-1))
    angle2 = torch.atan2(double_area, (e20 * e21).sum(dim=-1))

    vertex_area = pos.new_zeros((num_nodes,))
    face_area = (
        data.face_area.float()
        if hasattr(data, "face_area") and data.face_area is not None
        else _compute_face_area_from_geometry(pos, face)
    )
    face_area_thirds = (face_area / 3.0).to(pos.device)
    for vertex_idx in range(3):
        vertex_area = vertex_area + scatter(
            face_area_thirds,
            face[vertex_idx],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
    vertex_area = vertex_area.clamp_min(eps)

    angle_sum = pos.new_zeros((num_nodes,))
    angle_sum = angle_sum + scatter(angle0, face[0], dim=0, dim_size=num_nodes, reduce="sum")
    angle_sum = angle_sum + scatter(angle1, face[1], dim=0, dim_size=num_nodes, reduce="sum")
    angle_sum = angle_sum + scatter(angle2, face[2], dim=0, dim_size=num_nodes, reduce="sum")

    laplace_num = pos.new_zeros((num_nodes, 3))
    laplace_num.index_add_(0, face[0], cot2.unsqueeze(-1) * (v1 - v0) + cot1.unsqueeze(-1) * (v2 - v0))
    laplace_num.index_add_(0, face[1], cot0.unsqueeze(-1) * (v2 - v1) + cot2.unsqueeze(-1) * (v0 - v1))
    laplace_num.index_add_(0, face[2], cot1.unsqueeze(-1) * (v0 - v2) + cot0.unsqueeze(-1) * (v1 - v2))
    laplace = laplace_num / (2.0 * vertex_area.unsqueeze(-1))

    gaussian_curvature = ((2.0 * torch.pi) - angle_sum) / vertex_area

    if hasattr(data, "normal") and data.normal is not None:
        normal = torch.nn.functional.normalize(
            data.normal.float(), p=2, dim=-1, eps=eps
        )
        mean_curvature = -0.5 * (laplace * normal).sum(dim=-1)
    else:
        mean_curvature = 0.5 * torch.linalg.norm(laplace, dim=-1)

    discriminant = torch.clamp(mean_curvature.square() - gaussian_curvature, min=0.0)
    root = torch.sqrt(discriminant)
    k_min = mean_curvature - root
    k_max = mean_curvature + root

    gaussian_curvature = torch.nan_to_num(
        gaussian_curvature, nan=0.0, posinf=0.0, neginf=0.0
    )
    principal_curvatures = torch.stack(
        [
            torch.nan_to_num(k_min, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(k_max, nan=0.0, posinf=0.0, neginf=0.0),
        ],
        dim=-1,
    )

    gaussian_curvature = _safe_max_abs_normalize(
        gaussian_curvature.unsqueeze(-1), eps=eps
    )
    principal_curvatures = _safe_max_abs_normalize(principal_curvatures, eps=eps)

    return gaussian_curvature, principal_curvatures


def compute_thickness(
    data: Union[MeshData, Batch],
    d: float | None = None,
    k: int = 32,
    max_k: int = 96,
    eps: float = 1e-8,
):
    """Compute per-node thickness and representative paired-node indices.

    Definition:
      1) Query candidate vertices near the target point x_i - d * n_i.
      2) Keep only vertices behind the surface, i.e. (x_j - x_i) · n_i < 0.
      3) Compute a weighted average of projected depths along -n_i.
      4) Keep the highest-weight candidate as the representative pair T(v_i).

    Notes:
      - Supports PyG `Batch` (preferred) or single `MeshData`.
      - Assumes `data.pos` and `data.normal` are present.
      - If `d` is None, uses per-graph bounding-box diagonal length.

    Returns:
      thickness: (num_nodes,) float tensor
      paired_idx: (num_nodes,) long tensor mapping each node to its paired node
    """

    if not hasattr(data, "pos"):
        raise AttributeError("compute_thickness requires `data.pos`.")
    if not hasattr(data, "normal"):
        raise AttributeError("compute_thickness requires `data.normal`.")

    pos = data.pos
    normal = data.normal
    device = pos.device
    num_nodes = pos.size(0)

    if isinstance(data, Batch):
        batch = data.batch
    else:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Ensure unit normals (robust to non-normalized inputs)
    normal = torch.nn.functional.normalize(normal, p=2, dim=-1, eps=eps)

    if d is None:
        # Per-graph projection distance: bbox diagonal.
        pos_min = scatter(pos, batch, dim=0, reduce="min")
        pos_max = scatter(pos, batch, dim=0, reduce="max")
        diag = torch.linalg.norm(pos_max - pos_min, dim=-1).clamp_min(eps)
        d_node = diag[batch]
    else:
        d_node = torch.full((num_nodes,), float(d), dtype=pos.dtype, device=device)

    target = pos - d_node.unsqueeze(-1) * normal

    # Graph sizes (used for edge cases)
    ones = torch.ones((num_nodes,), dtype=pos.dtype, device=device)
    graph_counts = scatter(ones, batch, dim=0, reduce="sum").to(torch.long)
    small_graph_mask = graph_counts[batch] < 2

    thickness = torch.zeros((num_nodes,), dtype=pos.dtype, device=device)
    paired_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    unresolved = ~small_graph_mask

    # Fast path: batched kNN via torch_cluster + segment min via torch_scatter.
    # We increase k until all nodes resolve (or we hit max_k).
    big_value = torch.tensor(1e30, dtype=pos.dtype, device=device)

    k_cur = max(2, int(k))
    while unresolved.any() and k_cur <= int(max_k):
        # edge_index: [2, k_cur * num_nodes]
        edge_index = torch_cluster_knn(
            x=pos,
            y=target,
            k=min(k_cur, num_nodes),
            batch_x=batch,
            batch_y=batch,
        )
        y_idx = edge_index[0]
        x_idx = edge_index[1]

        vec = pos[x_idx] - pos[y_idx]
        target_dist = torch.linalg.norm(pos[x_idx] - target[y_idx], dim=-1)
        projected_depth = -(vec * normal[y_idx]).sum(dim=-1)

        invalid_edge = (x_idx == y_idx) | (projected_depth <= 0)
        safe_target_dist = target_dist.masked_fill(invalid_edge, big_value)

        valid_weight = torch.zeros_like(target_dist)
        valid_mask = ~invalid_edge
        valid_weight[valid_mask] = 1.0 / (target_dist[valid_mask] + eps)

        weight_sum = scatter(
            valid_weight,
            y_idx,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        weighted_depth_sum = scatter(
            valid_weight * projected_depth.clamp_min(0.0),
            y_idx,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        _, argmax_edge = scatter_max(
            valid_weight,
            y_idx,
            dim=0,
            dim_size=num_nodes,
        )

        cand_paired = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        valid_arg = argmax_edge >= 0
        cand_paired[valid_arg] = x_idx[argmax_edge[valid_arg]]

        cand_ok = weight_sum > 0
        newly_resolved = unresolved & cand_ok & (cand_paired >= 0)
        if newly_resolved.any():
            paired_idx[newly_resolved] = cand_paired[newly_resolved]
            thickness[newly_resolved] = (
                weighted_depth_sum[newly_resolved]
                / weight_sum[newly_resolved].clamp_min(eps)
            )
            unresolved = unresolved & ~newly_resolved

        k_cur *= 2

    # Fallback for any remaining unresolved nodes: nearest neighbor excluding self.
    if unresolved.any():
        # For safety, compute 2-NN; pick the one that's not self.
        edge_index = torch_cluster_knn(
            x=pos,
            y=pos,
            k=min(2, num_nodes),
            batch_x=batch,
            batch_y=batch,
        )
        y_idx = edge_index[0]
        x_idx = edge_index[1]

        dist = torch.linalg.norm(pos[x_idx] - pos[y_idx], dim=-1)
        dist = dist.masked_fill(x_idx == y_idx, big_value)
        min_dist, argmin_edge = scatter_min(dist, y_idx, dim=0, dim_size=num_nodes)
        valid = unresolved & (min_dist < (big_value * 0.5)) & (argmin_edge >= 0)
        if valid.any():
            paired = x_idx[argmin_edge[valid]]
            paired_idx[valid] = paired
            thickness[valid] = torch.linalg.norm(pos[valid] - pos[paired], dim=-1)

    # Handle size-1 graphs explicitly
    paired_idx[small_graph_mask] = torch.arange(num_nodes, device=device)[
        small_graph_mask
    ]
    thickness[small_graph_mask] = 0
    return thickness, paired_idx
