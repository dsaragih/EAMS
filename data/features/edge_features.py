import trimesh
from data.types import MeshData
from torch_geometric.data import Batch
from typing import Union
import torch
import numpy as np

from torch_geometric.utils import scatter


def _safe_max_abs_normalize(
    values: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if values.numel() == 0:
        return values
    if values.ndim == 1:
        scale = values.abs().max().clamp_min(eps)
        return values / scale
    scale = values.abs().amax(dim=0, keepdim=True).clamp_min(eps)
    return values / scale


def _bbox_center(pos: torch.Tensor) -> torch.Tensor:
    pos_min = pos.min(dim=0).values
    pos_max = pos.max(dim=0).values
    return 0.5 * (pos_min + pos_max)


def compute_edge_degree_weight(data: Union[MeshData, Batch]):
    idx_to, idx_from = data.edge_index
    num_neighbours = scatter(torch.ones_like(idx_from), index=idx_to, reduce="sum")
    return 1.0 / num_neighbours[idx_to]


def _resolve_edge_label_layout(
    data: Union[MeshData, Batch], edge_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_count = int(data.edge_index.shape[1])
    label_count = int(data.edge_labels.shape[0])

    if label_count == edge_count:
        return data.edge_labels, edge_lengths

    undirected_edges = torch.sort(data.edge_index, dim=0).values.T
    _, inverse = torch.unique(undirected_edges, dim=0, return_inverse=True)
    unique_edge_count = int(inverse.max().item() + 1) if inverse.numel() > 0 else 0

    if label_count != unique_edge_count:
        raise ValueError(
            "edge_labels must align with either directed edges or unique undirected edges: "
            f"got edge_labels={label_count}, directed_edges={edge_count}, unique_edges={unique_edge_count}."
        )

    # Both directions of an undirected edge have the same length. Reducing by mean
    # gives one representative length per supervised edge label.
    undirected_lengths = scatter(edge_lengths, inverse, dim=0, reduce="mean")
    return data.edge_labels, undirected_lengths


def compute_edge_anatomy_weight(data: Union[MeshData, Batch]):
    # Compute edge lengths
    edges = data.edge_index.T
    edge_lengths = torch.norm(data.pos[edges[:, 0]] - data.pos[edges[:, 1]], dim=1).to(
        data.pos.device
    )

    edge_labels, edge_lengths_for_labels = _resolve_edge_label_layout(
        data, edge_lengths
    )

    # Compute mean edge length per label (excluding background label)
    unique_labels = torch.unique(edge_labels)
    edge_weights = torch.ones(
        edge_lengths_for_labels.shape[0],
        device=data.pos.device,
        dtype=edge_lengths.dtype,
    )
    for label in unique_labels:
        if label != 0:
            mask = edge_labels == label
            label_mean = torch.mean(edge_lengths_for_labels[mask])
            edge_weights[mask] += edge_lengths_for_labels[mask] / label_mean

    if edge_weights.shape[0] != edge_lengths.shape[0]:
        undirected_edges = torch.sort(data.edge_index, dim=0).values.T
        _, inverse = torch.unique(undirected_edges, dim=0, return_inverse=True)
        edge_weights = edge_weights[inverse]

    return edge_weights.to(data.pos.device)


def copy_edge_weight(data: Union[MeshData, Batch]):
    # Assume that data.edge_weights is already computed and stored in the data object
    if hasattr(data, "edge_weights") and data.edge_weights is not None:
        return data.edge_weights.to(data.pos.device)
    elif hasattr(data, "edge_weight") and data.edge_weight is not None:
        return data.edge_weight.to(data.pos.device)
    else:
        # Should not fail, but just return tensor with fill value 1e-5
        return torch.full((data.edge_index.shape[1],), 1e-5, device=data.pos.device)


def compute_symmetric_ratios(data: Union[MeshData, Batch]):
    # Handle batch processing
    if isinstance(data, Batch):
        symmetric_ratios = []
        graphs = data.to_data_list()
        for graph in graphs:
            symmetric_ratio = _compute_symmetric_ratios_single(graph.cpu())
            symmetric_ratios.append(symmetric_ratio.to(data.pos.device))

        return torch.cat(symmetric_ratios, dim=0)

    # Handle single graph
    return _compute_symmetric_ratios_single(data)


def _compute_symmetric_ratios_single(data: MeshData):
    # Get edge lengths from trimesh
    edge_lengths = torch.norm(
        data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]], dim=1
    )

    # Initialize symmetric ratios
    num_edges = data.edge_index.shape[1]
    edge_symmetric_ratios = torch.full(
        (num_edges, 2), 1.0, dtype=torch.float32, device=data.pos.device
    )  # Default to 1.0

    if data.face_adjacency.shape[0] == 0:
        return edge_symmetric_ratios

    # Get edge indices for each adjacency pair
    edge_indices = data.face_adjacency_edge_index  # (n_adjacent_pairs,)

    # Get edge vertices for all relevant edges
    edge_verts = data.edge_index[:, edge_indices]  # (2, n_adjacent_pairs)

    # Get positions of edge vertices
    v1 = data.pos[edge_verts[0]]  # (n_adjacent_pairs, 3)
    v2 = data.pos[edge_verts[1]]  # (n_adjacent_pairs, 3)
    edge_vecs = v2 - v1  # (n_adjacent_pairs, 3)

    # Get edge lengths for the adjacency edges
    adj_edge_lengths = edge_lengths[edge_indices]  # (n_adjacent_pairs,)

    # Get face vertices for both faces in each pair
    face_verts_0 = data.face[
        :, data.face_adjacency[:, 0].long()
    ]  # (3, n_adjacent_pairs)
    face_verts_1 = data.face[
        :, data.face_adjacency[:, 1].long()
    ]  # (3, n_adjacent_pairs)

    # Find opposite vertices (vertex in face not in edge)
    # For each face, find which vertex is not in the edge
    edge_set_0 = edge_verts[0].unsqueeze(0)  # (1, n_adjacent_pairs)
    edge_set_1 = edge_verts[1].unsqueeze(0)  # (1, n_adjacent_pairs)

    # Check which vertices in face are not in edge for face 0
    mask_0_not_in_edge = (face_verts_0 != edge_set_0) & (
        face_verts_0 != edge_set_1
    )  # (3, n_adjacent_pairs)
    opposite_idx_0 = torch.argmax(
        mask_0_not_in_edge.float(), dim=0
    )  # (n_adjacent_pairs,)
    opposite_verts_0 = face_verts_0[opposite_idx_0, torch.arange(face_verts_0.shape[1])]

    # Check which vertices in face are not in edge for face 1
    mask_1_not_in_edge = (face_verts_1 != edge_set_0) & (
        face_verts_1 != edge_set_1
    )  # (3, n_adjacent_pairs)
    opposite_idx_1 = torch.argmax(
        mask_1_not_in_edge.float(), dim=0
    )  # (n_adjacent_pairs,)
    opposite_verts_1 = face_verts_1[opposite_idx_1, torch.arange(face_verts_1.shape[1])]

    # Compute heights for both faces
    for j, opposite_verts in enumerate([opposite_verts_0, opposite_verts_1]):
        v_opposite = data.pos[opposite_verts]  # (n_adjacent_pairs, 3)
        v1_to_opposite = v_opposite - v1  # (n_adjacent_pairs, 3)

        # Compute edge unit vectors
        edge_unit = edge_vecs / adj_edge_lengths.unsqueeze(-1)  # (n_adjacent_pairs, 3)

        # Project onto edge direction
        projection_length = (v1_to_opposite * edge_unit).sum(
            dim=1
        )  # (n_adjacent_pairs,)

        # Perpendicular component
        perp_vec = (
            v1_to_opposite - projection_length.unsqueeze(-1) * edge_unit
        )  # (n_adjacent_pairs, 3)
        heights = torch.norm(perp_vec, dim=1)  # (n_adjacent_pairs,)

        # Compute ratios
        ratios = heights / (adj_edge_lengths + 1e-10)  # (n_adjacent_pairs,)

        # Assign to output
        edge_symmetric_ratios[edge_indices, j] = ratios

    return edge_symmetric_ratios


def compute_symmetric_angles(data: Union[MeshData, Batch]):
    # Handle batch processing
    if isinstance(data, Batch):
        symmetric_angles = []
        graphs = data.to_data_list()

        for graph in graphs:
            symmetric_angle = _compute_symmetric_angles_single(graph.cpu())
            symmetric_angles.append(symmetric_angle.to(data.pos.device))

        return torch.cat(symmetric_angles, dim=0)

    # Handle single graph
    return _compute_symmetric_angles_single(data)


def _compute_symmetric_angles_single(data: Union[MeshData, Batch]):
    mesh = trimesh.Trimesh(
        vertices=data.pos.cpu(), faces=data.face.T.cpu(), process=False
    )
    face_angles = torch.tensor(mesh.face_angles, dtype=torch.float32)  # (n_faces, 3)
    num_edges = data.edge_index.shape[1]
    edge_opposite_angles = torch.full(
        (num_edges, 2), np.pi / 3, dtype=torch.float32, device=data.pos.device
    )  # Default to 60° for equilateral

    if data.face_adjacency.shape[0] == 0:
        return edge_opposite_angles.to(data.pos.device)

    # Get edge indices for each adjacency pair
    edge_indices = data.face_adjacency_edge_index  # (n_adjacent_pairs,)

    # Get edge vertices for all relevant edges
    edge_verts = data.edge_index[:, edge_indices]  # (2, n_adjacent_pairs)

    # Get face indices
    face_idx_0 = data.face_adjacency[:, 0].long()  # (n_adjacent_pairs,)
    face_idx_1 = data.face_adjacency[:, 1].long()  # (n_adjacent_pairs,)

    # Get face vertices for both faces in each pair
    face_verts_0 = data.face[:, face_idx_0]  # (3, n_adjacent_pairs)
    face_verts_1 = data.face[:, face_idx_1]  # (3, n_adjacent_pairs)

    # Find opposite vertices (vertex in face not in edge)
    edge_set_0 = edge_verts[0].unsqueeze(0)  # (1, n_adjacent_pairs)
    edge_set_1 = edge_verts[1].unsqueeze(0)  # (1, n_adjacent_pairs)

    # For face 0: find which vertex is not in edge
    mask_0_not_in_edge = (face_verts_0 != edge_set_0) & (
        face_verts_0 != edge_set_1
    )  # (3, n_adjacent_pairs)
    opposite_local_idx_0 = torch.argmax(
        mask_0_not_in_edge.float(), dim=0
    )  # (n_adjacent_pairs,)

    # For face 1: find which vertex is not in edge
    mask_1_not_in_edge = (face_verts_1 != edge_set_0) & (
        face_verts_1 != edge_set_1
    )  # (3, n_adjacent_pairs)
    opposite_local_idx_1 = torch.argmax(
        mask_1_not_in_edge.float(), dim=0
    )  # (n_adjacent_pairs,)

    # Get the angles at opposite vertices
    angles_0 = face_angles[face_idx_0, opposite_local_idx_0]  # (n_adjacent_pairs,)
    angles_1 = face_angles[face_idx_1, opposite_local_idx_1]  # (n_adjacent_pairs,)

    # Assign to output
    edge_opposite_angles[edge_indices, 0] = angles_0
    edge_opposite_angles[edge_indices, 1] = angles_1

    return edge_opposite_angles


def compute_dihedrals(data: Union[MeshData, Batch]):
    # Handle batch processing
    if isinstance(data, Batch):
        dihedrals = []
        graphs = data.to_data_list()

        for graph in graphs:
            dihedral = _compute_dihedrals_single(graph.cpu())
            dihedrals.append(dihedral.to(data.pos.device))

        return torch.cat(dihedrals, dim=0)

    # Handle single graph
    return _compute_dihedrals_single(data)


def _compute_dihedrals_single(data: Union[MeshData, Batch]):
    mesh = trimesh.Trimesh(
        vertices=data.pos.cpu(), faces=data.face.T.cpu(), process=False
    )
    dihedrals = torch.tensor(
        mesh.face_adjacency_angles, dtype=torch.float32, device=data.pos.device
    )  # (n_adjacent_pairs,)
    edge_dihedral_angles = torch.full(
        (data.edge_index.shape[1],), np.pi, dtype=torch.float32, device=data.pos.device
    )

    if data.face_adjacency.shape[0] == 0:
        return edge_dihedral_angles

    # Vectorized assignment: map dihedral angles to their corresponding edges
    edge_indices = data.face_adjacency_edge_index  # (n_adjacent_pairs,)
    edge_dihedral_angles[edge_indices] = dihedrals

    return edge_dihedral_angles


def compute_signed_dihedrals(data: Union[MeshData, Batch]):
    if isinstance(data, Batch):
        signed_dihedrals = []
        graphs = data.to_data_list()

        for graph in graphs:
            signed = _compute_signed_dihedrals_single(graph.cpu())
            signed_dihedrals.append(signed.to(data.pos.device))

        return torch.cat(signed_dihedrals, dim=0)

    return _compute_signed_dihedrals_single(data)


def compute_chiral_edge_volume(data: Union[MeshData, Batch]):
    if isinstance(data, Batch):
        chiral_values = []
        graphs = data.to_data_list()

        for graph in graphs:
            chiral = _compute_chiral_edge_volume_single(graph.cpu())
            chiral_values.append(chiral.to(data.pos.device))

        return torch.cat(chiral_values, dim=0)

    return _compute_chiral_edge_volume_single(data)


def compute_edge_vertical_alignment(data: Union[MeshData, Batch]):
    if isinstance(data, Batch):
        values = []
        for graph in data.to_data_list():
            values.append(_compute_edge_vertical_alignment_single(graph.cpu()))
        return torch.cat(values, dim=0).to(data.pos.device)

    return _compute_edge_vertical_alignment_single(data)


def compute_edge_midline_proximity(
    data: Union[MeshData, Batch],
    eps: float = 1e-12,
):
    if isinstance(data, Batch):
        values = []
        for graph in data.to_data_list():
            values.append(_compute_edge_midline_proximity_single(graph.cpu(), eps=eps))
        return torch.cat(values, dim=0).to(data.pos.device)

    return _compute_edge_midline_proximity_single(data, eps=eps)


def _compute_edge_vertical_alignment_single(data: MeshData):
    if data.edge_index.numel() == 0:
        return data.pos.new_zeros((0,))

    pos = data.pos.float()
    edge_vertices = data.edge_index.long()
    edge_vec = pos[edge_vertices[1]] - pos[edge_vertices[0]]
    edge_dir = torch.nn.functional.normalize(edge_vec, p=2, dim=-1, eps=1e-12)
    return edge_dir[:, 2].abs().clamp(max=1.0)


def _compute_edge_midline_proximity_single(
    data: MeshData,
    eps: float = 1e-12,
):
    if data.edge_index.numel() == 0:
        return data.pos.new_zeros((0,))

    pos = data.pos.float()
    center = _bbox_center(pos)
    x = pos[:, 0]
    x_min = x.min()
    x_max = x.max()
    half_extent = (0.5 * (x_max - x_min)).clamp_min(eps)

    edge_vertices = data.edge_index.long()
    midpoints_x = 0.5 * (x[edge_vertices[0]] + x[edge_vertices[1]])
    distance = (midpoints_x - center[0]).abs() / half_extent
    proximity = 1.0 - distance.clamp(max=1.0)
    return proximity


def _adjacent_face_geometry_single(
    data: MeshData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if data.face_adjacency.shape[0] == 0:
        return None

    edge_indices = data.face_adjacency_edge_index.long()
    valid = edge_indices >= 0
    if not valid.any():
        return None

    face_pairs = data.face_adjacency[valid].long()
    edge_indices = edge_indices[valid]

    face = data.face.long()
    pos = data.pos.float()
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    face_normals = torch.nn.functional.normalize(
        torch.cross(v1 - v0, v2 - v0, dim=-1),
        p=2,
        dim=-1,
        eps=1e-12,
    )

    n0 = face_normals[face_pairs[:, 0]]
    n1 = face_normals[face_pairs[:, 1]]

    edge_vertices = data.edge_index[:, edge_indices].long()
    edge_vec = pos[edge_vertices[1]] - pos[edge_vertices[0]]
    return edge_indices, edge_vec, n0, n1


def _compute_signed_dihedrals_single(data: MeshData):
    signed_dihedral_angles = torch.zeros(
        (data.edge_index.shape[1],), dtype=torch.float32, device=data.pos.device
    )

    geometry = _adjacent_face_geometry_single(data)
    if geometry is None:
        return signed_dihedral_angles

    edge_indices, edge_vec, n0, n1 = geometry
    dot = torch.clamp((n0 * n1).sum(dim=-1), min=-1.0, max=1.0)
    angle = torch.acos(dot)
    edge_dir = torch.nn.functional.normalize(edge_vec, p=2, dim=-1, eps=1e-12)
    orientation = (edge_dir * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    sign = torch.where(orientation < 0.0, -1.0, 1.0)
    signed_dihedral_angles[edge_indices] = (angle / torch.pi) * sign
    return signed_dihedral_angles


def _compute_chiral_edge_volume_single(data: MeshData):
    chiral_values = torch.zeros(
        (data.edge_index.shape[1],), dtype=torch.float32, device=data.pos.device
    )

    geometry = _adjacent_face_geometry_single(data)
    if geometry is None:
        return chiral_values

    edge_indices, edge_vec, n0, n1 = geometry
    chiral_values[edge_indices] = (edge_vec * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    return _safe_max_abs_normalize(chiral_values)
