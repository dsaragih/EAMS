import torch
from torch_geometric.data import Data
import numpy as np
import trimesh
import pymeshlab as ml
from scipy.spatial import cKDTree
import torch_geometric.transforms as T
from .signature import (
    DEFAULT_SIGNATURE_APPROX,
    DEFAULT_SIGNATURE_BASIS,
    get_signature_cache_path,
    load_or_build_signature_extractor,
)

from loguru import logger


def _as_face_f3(face: torch.Tensor) -> torch.Tensor:
    """Return faces as (F, 3) long tensor."""
    if face is None:
        return torch.empty((0, 3), dtype=torch.long)
    if not isinstance(face, torch.Tensor):
        face = torch.as_tensor(face)
    if face.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=face.device)
    if face.ndim != 2:
        raise ValueError(f"Expected face to be 2D, got shape={tuple(face.shape)}")
    # Common PyG mesh convention: (3, F)
    if face.shape[0] == 3 and face.shape[1] != 3:
        return face.t().contiguous().to(torch.long)
    if face.shape[1] == 3:
        return face.contiguous().to(torch.long)
    raise ValueError(
        f"Unrecognized face shape={tuple(face.shape)}; expected (3,F) or (F,3)"
    )


def _build_edge_index_from_faces(face_f3: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build bidirectional, unique edge_index from triangular faces."""
    if face_f3.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=face_f3.device)
    a, b, c = face_f3[:, 0], face_f3[:, 1], face_f3[:, 2]
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


def _remap_edge_labels_to_face_edges(
    previous_edge_index: torch.Tensor | None,
    previous_edge_labels: torch.Tensor | None,
    old_to_new: torch.Tensor,
    current_edge_index: torch.Tensor,
) -> torch.Tensor | None:
    """Map old edge labels onto rebuilt edges by canonical undirected edge tuples."""
    if previous_edge_index is None or previous_edge_labels is None:
        return None
    if current_edge_index.numel() == 0:
        return torch.empty(
            (0,), dtype=previous_edge_labels.dtype, device=old_to_new.device
        )

    warn_unmapped_ratio = 0.05
    previous_edge_index = previous_edge_index.to(
        device=old_to_new.device, dtype=torch.long
    )
    previous_edge_labels = previous_edge_labels.to(device=old_to_new.device)

    label_by_edge: dict[tuple[int, int], int] = {}
    conflict_count = 0
    directed_edge_count = int(previous_edge_index.shape[1])

    if int(previous_edge_labels.shape[0]) == directed_edge_count:
        remapped_previous = old_to_new[previous_edge_index]
        keep_mask = (
            (remapped_previous[0] >= 0)
            & (remapped_previous[1] >= 0)
            & (remapped_previous[0] != remapped_previous[1])
        )
        canonical_previous = (
            torch.sort(remapped_previous[:, keep_mask], dim=0).values.t().contiguous()
        )
        label_values = previous_edge_labels[keep_mask]
    else:
        previous_unique_edges = (
            torch.sort(previous_edge_index, dim=0).values.t().unique(dim=0)
        )
        if int(previous_edge_labels.shape[0]) != int(previous_unique_edges.shape[0]):
            raise ValueError(
                "edge_labels must align with either directed edges or unique undirected edges: "
                f"got labels={previous_edge_labels.shape[0]}, "
                f"directed_edges={directed_edge_count}, "
                f"unique_edges={previous_unique_edges.shape[0]}."
            )

        keep_mask = (old_to_new[previous_unique_edges[:, 0]] >= 0) & (
            old_to_new[previous_unique_edges[:, 1]] >= 0
        )
        canonical_previous = torch.stack(
            [
                old_to_new[previous_unique_edges[keep_mask, 0]],
                old_to_new[previous_unique_edges[keep_mask, 1]],
            ],
            dim=1,
        )
        non_self_mask = canonical_previous[:, 0] != canonical_previous[:, 1]
        canonical_previous = canonical_previous[non_self_mask]
        label_values = previous_edge_labels[keep_mask][non_self_mask]

    for edge_tuple, label_value in zip(
        canonical_previous.tolist(), label_values.tolist()
    ):
        key = (int(edge_tuple[0]), int(edge_tuple[1]))
        existing_value = label_by_edge.get(key)
        if existing_value is None:
            label_by_edge[key] = int(label_value)
        elif existing_value != int(label_value):
            conflict_count += 1

    if conflict_count > 0:
        logger.warning(
            "Found {} conflicting labels while collapsing old directed edges to undirected tuples; keeping first-seen labels.",
            conflict_count,
        )

    current_canonical = (
        torch.sort(current_edge_index.to(device=old_to_new.device), dim=0)
        .values.t()
        .contiguous()
    )
    remapped = torch.full(
        (current_canonical.shape[0],),
        -1,
        dtype=previous_edge_labels.dtype,
        device=old_to_new.device,
    )
    for idx, edge_tuple in enumerate(current_canonical.tolist()):
        key = (int(edge_tuple[0]), int(edge_tuple[1]))
        if key in label_by_edge:
            remapped[idx] = label_by_edge[key]

    unmapped_count = int((remapped < 0).sum().item())
    total_count = int(remapped.shape[0])
    if total_count > 0 and (unmapped_count / total_count) > warn_unmapped_ratio:
        logger.warning(
            "Edge label remap left {}/{} rebuilt edges unmapped ({:.2f}%).",
            unmapped_count,
            total_count,
            100.0 * unmapped_count / total_count,
        )

    return remapped


def _subset_vertex_level_tensors(
    data: Data,
    vertex_indices: torch.Tensor,
    *,
    excluded_keys: tuple[str, ...] = ("pos", "face", "edge_index", "edge_labels"),
) -> None:
    if vertex_indices.ndim != 1:
        raise ValueError(
            f"Expected 1D vertex_indices, got shape={tuple(vertex_indices.shape)}"
        )

    vertex_count_before = int(data.pos.shape[0])
    for key in list(data.keys()):
        if key in excluded_keys:
            continue
        value = data[key]
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == vertex_count_before
        ):
            data[key] = value[vertex_indices]


def _subset_face_level_tensors(
    data: Data,
    face_keep: torch.Tensor,
    *,
    allowed_keys: tuple[str, ...] = ("face_labels", "face_area"),
) -> None:
    if face_keep.ndim != 1:
        raise ValueError(
            f"Expected 1D face_keep mask, got shape={tuple(face_keep.shape)}"
        )

    face_count_before = int(face_keep.shape[0])
    for key in allowed_keys:
        if not hasattr(data, key):
            continue
        value = getattr(data, key)
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == face_count_before
        ):
            setattr(data, key, value[face_keep])


# Can add stuff like non-manifold edges, boundary edges, etc.
def compose_graph_pretransforms(transforms: list[str]) -> T.Compose:
    """Compose an iterable of Transforms into a single transform.

    :param pretransforms: An iterable of pretransforms.
    :type pretransforms: Iterable[Callable]
    :raises ValueError: If ``pretransforms`` is not a list or dict.
    :return: A single transform.
    :rtype: T.Compose
    """
    TRANSFORM_FACTORY = {
        "mesh_cleanup": compute_mesh_cleanup,
        "normalized_coords": compute_normalized_coords,
        "normals": compute_normals_from_mesh,
        "on_manifold_face_info": compute_on_manifold_face_info,
        "hks": compute_hks,
        "signature_spectrum": compute_signature_spectrum,
        "face_adjacency": compute_face_adjacency,
        "face_adjacency_edge_index": compute_face_adjacency_edge_index,
        "vertex_to_edge": compute_vertex_to_edge,
    }
    callable_list = []

    # Ensure cleanup runs first if requested (prevents downstream spectral ops from
    # failing on degenerate meshes).
    if "mesh_cleanup" in transforms and (
        len(transforms) == 0 or transforms[0] != "mesh_cleanup"
    ):
        logger.warning(
            "Reordering pre_transforms to run 'mesh_cleanup' first (requested transforms: %s)",
            transforms,
        )
        transforms = ["mesh_cleanup"] + [t for t in transforms if t != "mesh_cleanup"]

    for transform in transforms:
        if transform in TRANSFORM_FACTORY:
            callable_list.append(TRANSFORM_FACTORY[transform])
        # If compute hierarchy, expect "compute_hierarchy:levels:keep_fraction"
        elif transform.startswith("compute_hierarchy"):
            parts = transform.split(":")
            levels = int(parts[1]) if len(parts) > 1 else 3
            keep_fraction = float(parts[2]) if len(parts) > 2 else 0.5
            callable_list.append(
                ComputeHierarchy(levels=levels, keep_fraction=keep_fraction)
            )
        else:
            raise ValueError(f"Transform '{transform}' not recognized.")

    logger.info(f"Composed pre-transforms: {transforms}")
    return T.Compose(callable_list)


def compute_mesh_cleanup(
    data: Data,
    area_eps: float = 1e-12,
    keep_min_faces: int = 4,
) -> Data:
    """Clean mesh geometry to make downstream ops (normals, Laplacian/HKS) stable.

    Operations (deterministic):
    - Drop non-finite vertices and any faces touching them
    - Drop degenerate faces (repeated indices) and near-zero-area faces
    - Remove unreferenced vertices and reindex faces
    - Rebuild edge_index from faces
    - Remap old edge labels onto rebuilt edges by canonical undirected edge tuple
    - Subset any per-vertex tensors (e.g., vertex_labels, norm/normal) accordingly

    Notes:
    - We intentionally avoid vertex merging / remeshing here to keep label alignment.
    - If the mesh becomes too small, we keep it but downstream spectral features should fall back.
    """

    if not hasattr(data, "pos") or data.pos is None:
        return data
    if not hasattr(data, "face") or data.face is None:
        return data

    previous_edge_index = getattr(data, "edge_index", None)
    previous_edge_labels = getattr(data, "edge_labels", None)
    pos: torch.Tensor = data.pos
    face_f3 = _as_face_f3(data.face).to(device=pos.device)

    num_vertices_old = int(pos.shape[0])
    if num_vertices_old == 0:
        return data

    active_original_indices = torch.arange(num_vertices_old, device=pos.device)

    # 1) Remove non-finite vertices
    finite_mask = torch.isfinite(pos).all(dim=1)
    if not finite_mask.all():
        valid_idx = torch.nonzero(finite_mask, as_tuple=False).view(-1)
        if valid_idx.numel() == 0:
            logger.warning(
                "Mesh cleanup: all vertices are non-finite; clearing faces/edges"
            )
            data.face = torch.empty((3, 0), dtype=torch.long, device=pos.device)
            data.edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            if previous_edge_labels is not None:
                data.edge_labels = torch.empty(
                    (0,), dtype=previous_edge_labels.dtype, device=pos.device
                )
            data.pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            return data

        old_to_tmp = torch.full(
            (num_vertices_old,), -1, dtype=torch.long, device=pos.device
        )
        old_to_tmp[valid_idx] = torch.arange(valid_idx.numel(), device=pos.device)
        face_keep = (old_to_tmp[face_f3] >= 0).all(dim=1)
        if not face_keep.all():
            _subset_face_level_tensors(data, face_keep)
        face_f3 = old_to_tmp[face_f3[face_keep]]

        _subset_vertex_level_tensors(data, valid_idx)
        pos = pos[valid_idx]
        active_original_indices = active_original_indices[valid_idx]

    # 2) Remove degenerate and near-zero-area faces
    if face_f3.numel() > 0:
        # repeated vertex indices
        not_repeated = (
            (face_f3[:, 0] != face_f3[:, 1])
            & (face_f3[:, 1] != face_f3[:, 2])
            & (face_f3[:, 0] != face_f3[:, 2])
        )

        # area check
        v0 = pos[face_f3[:, 0]]
        v1 = pos[face_f3[:, 1]]
        v2 = pos[face_f3[:, 2]]
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        double_area = torch.linalg.norm(cross, dim=1)
        nonzero_area = double_area > float(area_eps)

        face_keep = not_repeated & nonzero_area & torch.isfinite(double_area)
        if not face_keep.all():
            _subset_face_level_tensors(data, face_keep)
            face_f3 = face_f3[face_keep]

    # 3) Remove unreferenced vertices (and reindex faces)
    if face_f3.numel() > 0:
        used = torch.unique(face_f3.reshape(-1))
        used_sorted, _ = torch.sort(used)
        current_to_new = torch.full(
            (pos.shape[0],), -1, dtype=torch.long, device=pos.device
        )
        current_to_new[used_sorted] = torch.arange(
            used_sorted.numel(), device=pos.device
        )
        face_f3 = current_to_new[face_f3]

        _subset_vertex_level_tensors(data, used_sorted)
        pos = pos[used_sorted]
        active_original_indices = active_original_indices[used_sorted]
    else:
        # No faces => can't build a Laplacian/HKS meaningfully
        logger.warning("Mesh cleanup: mesh has 0 valid faces after filtering")

    # 4) Write back and rebuild edge_index
    data.pos = pos
    data.face = (
        face_f3.t().contiguous()
        if face_f3.numel() > 0
        else torch.empty((3, 0), dtype=torch.long, device=pos.device)
    )

    original_to_new = torch.full(
        (num_vertices_old,), -1, dtype=torch.long, device=pos.device
    )
    original_to_new[active_original_indices] = torch.arange(
        active_original_indices.numel(), device=pos.device
    )

    if face_f3.numel() > 0:
        data.edge_index = _build_edge_index_from_faces(
            face_f3, num_nodes=int(pos.shape[0])
        )
        remapped_edge_labels = _remap_edge_labels_to_face_edges(
            previous_edge_index=previous_edge_index,
            previous_edge_labels=previous_edge_labels,
            old_to_new=original_to_new,
            current_edge_index=data.edge_index,
        )
        if remapped_edge_labels is not None:
            data.edge_labels = remapped_edge_labels
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
        if previous_edge_labels is not None:
            data.edge_labels = torch.empty(
                (0,), dtype=previous_edge_labels.dtype, device=pos.device
            )

    # Warn if too small (helps explain HKS failures)
    if face_f3.shape[0] < int(keep_min_faces):
        logger.warning(
            "Mesh cleanup: very few faces remaining (%d); spectral features may be unstable",
            int(face_f3.shape[0]),
        )

    return data


def compute_normalized_coords(data):
    # Center and isotropically scale the mesh to fit in a unit cube around the origin
    center = data.pos.mean(dim=0)
    data.pos -= center
    bbox_diagonal = (data.pos.max(dim=0)[0] - data.pos.min(dim=0)[0]).norm()
    data.pos /= bbox_diagonal + 1e-12
    return data


def compute_normals_from_mesh(data):
    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )
    data.normal = torch.tensor(
        mesh.vertex_normals.copy(), dtype=data.pos.dtype, device=data.pos.device
    )
    return data


def compute_hks(data):
    logger.warning(
        "Pre-transform 'hks' is deprecated; use 'signature_spectrum' to cache eigensystems and compute signatures in FeatureFactory."
    )
    return compute_signature_spectrum(data)


def compute_signature_spectrum(data: Data):
    data.signature_approx = DEFAULT_SIGNATURE_APPROX
    data.signature_basis_count = DEFAULT_SIGNATURE_BASIS

    try:
        load_or_build_signature_extractor(
            data,
            approx=DEFAULT_SIGNATURE_APPROX,
            n_basis=DEFAULT_SIGNATURE_BASIS,
            cache_if_possible=True,
        )
        cache_path = get_signature_cache_path(
            data,
            approx=DEFAULT_SIGNATURE_APPROX,
            n_basis=DEFAULT_SIGNATURE_BASIS,
        )
        if cache_path is not None:
            data.signature_cache_path = str(cache_path)
    except Exception as e:
        logger.warning(
            "Signature spectrum caching failed (%s: %s); downstream features will recompute on demand.",
            type(e).__name__,
            e,
        )

    return data


def compute_face_adjacency(data: Data):
    # Handle single graph
    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.T.numpy(), process=False
    )

    data.face_adjacency = torch.tensor(
        mesh.face_adjacency, dtype=torch.long, device=data.pos.device
    )
    return data


def compute_face_adjacency_edge_index(data: Data):
    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.T.numpy(), process=False
    )
    edge_to_idx = {}
    for idx, edge in enumerate(data.edge_index.T.numpy()):
        edge_key = tuple(sorted(edge))
        edge_to_idx[edge_key] = idx

    face_to_edge_mapping = torch.full(
        (data.face_adjacency.shape[0],), -1, dtype=torch.long, device=data.pos.device
    )
    for i, edge_verts in enumerate(mesh.face_adjacency_edges):
        edge_key = tuple(sorted(edge_verts))
        if edge_key in edge_to_idx:
            face_to_edge_mapping[i] = edge_to_idx[edge_key]

    data.face_adjacency_edge_index = face_to_edge_mapping
    return data


def compute_vertex_to_edge(data: Data):
    """Helper function to compute vertex to edge mapping for a single graph."""
    # Compute a tensor of shape (num_edges*2, 2) of values (vertex_index, edge_index)
    edge_indices = torch.arange(data.edge_index.shape[1], device=data.edge_index.device)

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


def compute_on_manifold_face_info(data: Data):
    """
    Compute area and remove non-manifold faces from the mesh data.
    """
    assert data.face is not None, "Data object must have 'face' attribute."
    previous_edge_index = getattr(data, "edge_index", None)
    previous_edge_labels = getattr(data, "edge_labels", None)
    edges_set = set()
    faces = data.face.numpy().T  # (F, 3)
    mask = np.ones(len(faces), dtype=bool)

    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )
    data.face_area = torch.tensor(
        mesh.area_faces, dtype=data.pos.dtype, device=data.pos.device
    ).float()
    data.face_len = data.face.numpy().T.shape[0]

    face_areas = data.face_area.numpy()
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)

    _subset_face_level_tensors(
        data,
        torch.from_numpy(mask).to(device=data.pos.device),
        allowed_keys=("face_labels",),
    )
    data.face = data.face[:, mask]
    data.face_area = data.face_area[mask]
    data.face_len = data.face.numpy().T.shape[0]

    # IMPORTANT: face filtering can leave isolated/unreferenced vertices behind.
    # The HKS Laplacian builder constructs a diagonal mass matrix from incident
    # face areas; isolated vertices get zero mass, which makes (W - sigma M)
    # exactly singular and triggers "Factor is exactly singular".
    # Reindex/remove unreferenced vertices deterministically and rebuild edge_index.
    if data.face.numel() > 0:
        face_f3 = _as_face_f3(data.face).to(device=data.pos.device)
        used = torch.unique(face_f3.reshape(-1))
        used_sorted, _ = torch.sort(used)

        old_to_new = torch.full(
            (data.pos.shape[0],), -1, dtype=torch.long, device=data.pos.device
        )
        old_to_new[used_sorted] = torch.arange(
            used_sorted.numel(), device=data.pos.device
        )

        # Reindex faces
        face_f3 = old_to_new[face_f3]
        data.face = face_f3.t().contiguous()

        _subset_vertex_level_tensors(
            data,
            used_sorted,
            excluded_keys=(
                "pos",
                "face",
                "edge_index",
                "edge_labels",
                "face_area",
                "face_len",
            ),
        )

        data.pos = data.pos[used_sorted]

        # Rebuild edge_index from faces
        data.edge_index = _build_edge_index_from_faces(
            face_f3, num_nodes=int(data.pos.shape[0])
        )

        remapped_edge_labels = _remap_edge_labels_to_face_edges(
            previous_edge_index=previous_edge_index,
            previous_edge_labels=previous_edge_labels,
            old_to_new=old_to_new,
            current_edge_index=data.edge_index,
        )
        if remapped_edge_labels is not None:
            data.edge_labels = remapped_edge_labels
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long, device=data.pos.device)
        if previous_edge_labels is not None:
            data.edge_labels = torch.empty(
                (0,), dtype=previous_edge_labels.dtype, device=data.pos.device
            )
    return data


class ComputeHierarchy:
    def __init__(
        self,
        is_closed_manifold_oracle=None,
        levels: int = 3,
        keep_fraction: float = 0.5,
    ):
        self.is_closed_manifold_oracle = is_closed_manifold_oracle
        self.levels = levels
        self.keep_fraction = keep_fraction

    def __call__(self, data: Data) -> Data:
        """
        Build a multi-resolution mesh hierarchy by repeatedly simplifying the mesh and
        store fine->coarse correspondence mappings.

        Outputs (added to `data`)
        -------------------------
        data.pos_0 : torch.FloatTensor, shape (V0, 3)
            Vertex positions of the finest level (input mesh).
        data.face_0 : torch.LongTensor, shape (3, F0)
            Faces of the finest level (input mesh).
        data.pos_i : torch.FloatTensor, shape (Vi, 3)
            Vertex positions of level i coarse mesh.
        data.face_i : torch.LongTensor, shape (3, Fi)
            Faces of level i coarse mesh.
        data.fine_to_coarse_i : torch.LongTensor, shape (V_fine,)
            For level i, maps each fine vertex index to its nearest coarse vertex index.
        data.num_levels : int
            Number of levels in the hierarchy.


        Notes
        -----
        - For open meshes (holes/boundaries), we set preserveboundary=True to protect hole rims.
        """

        device = data.pos.device

        def _simplify(
            vertices: np.ndarray,
            faces_f3: np.ndarray,
            target_faces: int,
            is_closed: bool,
        ):
            # faces input is (F,3)
            ms = ml.MeshSet()
            ms.add_mesh(
                ml.Mesh(
                    vertex_matrix=vertices.astype(np.float64),
                    face_matrix=faces_f3.astype(np.int32),
                ),
                "m",
            )

            ms.compute_normal_per_face()
            ms.compute_normal_per_vertex()

            preserve_boundary = not bool(is_closed)
            preserve_topology = True  # anatomical-friendly default

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
            f2 = m2.face_matrix().astype(np.int32)  # (F2,3)
            return v2, f2

        def _fine_to_coarse_map(fine_v: np.ndarray, coarse_v: np.ndarray):
            tree = cKDTree(coarse_v)
            # workers=1 for more reproducible behavior across runs
            dist, idx = tree.query(fine_v, k=2, workers=1)
            return idx.astype(np.int64), dist.astype(np.float64)

        def _invert_map(fine_to_coarse: np.ndarray, num_coarse: int):
            order = np.argsort(fine_to_coarse, kind="stable")
            sorted_coarse = fine_to_coarse[order]
            counts = np.bincount(sorted_coarse, minlength=num_coarse)

            indptr = np.empty(num_coarse + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])

            indices = order.astype(np.int64)
            return indptr, indices

        # level 0 mesh (fine)
        pos0 = data.pos.detach().cpu().numpy().astype(np.float64)  # (V0,3)
        face0_f3 = data.face.detach().cpu().numpy().T.astype(np.int32)  # (F0,3)

        data.pos_0 = data.pos
        data.face_0 = data.face
        # delete data.pos
        # delete data.face
        del data.pos
        del data.face

        if self.is_closed_manifold_oracle is None:
            is_closed0 = False
        else:
            is_closed0 = bool(self.is_closed_manifold_oracle(pos0, face0_f3))

        prev_pos = pos0
        prev_faces_f3 = face0_f3
        is_closed = is_closed0

        for i in range(int(self.levels) - 1):
            target_faces = int(
                max(4, int(prev_faces_f3.shape[0] * float(self.keep_fraction)))
            )
            coarse_pos, coarse_faces_f3 = _simplify(
                prev_pos, prev_faces_f3, target_faces, is_closed
            )

            # fine -> coarse nearest neighbor map (vertex correspondence)
            fine_to_coarse, dist = _fine_to_coarse_map(prev_pos, coarse_pos)
            # indptr, indices = _invert_map(fine_to_coarse, num_coarse=coarse_pos.shape[0])

            setattr(data, f"pos_{i+1}", torch.from_numpy(coarse_pos).float().to(device))
            setattr(
                data,
                f"face_{i+1}",
                torch.from_numpy(coarse_faces_f3.T).long().to(device),
            )
            setattr(
                data,
                f"fine_to_coarse_{i}",
                torch.from_numpy(fine_to_coarse).long().to(device),
            )

            # next iteration
            prev_pos, prev_faces_f3 = coarse_pos, coarse_faces_f3
            # (optional) you could re-evaluate closure each level; simplest is to keep initial flag
            # is_closed = bool(is_closed_manifold_oracle(prev_pos, prev_faces_f3)) if oracle else is_closed

        data.num_levels = int(self.levels)
        return data
