import numpy as np
import os
import ntpath
from dataclasses import dataclass, field
from typing import Optional, List
from numpy.typing import NDArray
import random


@dataclass
class MeshData:
    """Mesh data structure with standardized interface.

    This class stores all mesh-related data including vertices, edges,
    connectivity information, and extracted features.
    """

    # Core mesh data
    vs: Optional[NDArray[np.float32]] = None  # Vertices (#V x 3)
    edges: Optional[NDArray[np.int32]] = None  # Edges (#E x 2)

    # GEMM (Graph Edge Mesh Model) connectivity
    gemm_edges: Optional[NDArray[np.int64]] = None  # One-ring neighbors (#E x 4)
    gemm_all: Optional[NDArray[np.int64]] = (
        None  # All neighboring edges (#E x 10) cropped/padded
    )
    sides: Optional[NDArray[np.int64]] = None  # Side indices (#E x 4)

    # Edge properties
    edges_count: int = 0  # Total number of edges
    edge_lengths: Optional[NDArray[np.float32]] = None  # Edge lengths (#E)
    edge_areas: Optional[NDArray[np.float32]] = None  # Edge areas (#E)

    # Vertex properties
    ve: Optional[List[List[int]]] = None  # Vertex to edges mapping
    v_mask: Optional[NDArray[np.bool_]] = None  # Vertex mask (#V)

    # Features
    features: Optional[NDArray[np.float32]] = (
        None  # Extracted features (#features x #E)
    )

    # File info
    filename: str = "unknown"  # Base filename
    fullfilename: str = ""  # Full path to file

    # Additional metadata
    shifted: float = 0.0  # Proportion of vertices shifted (for augmentation)

    def __getitem__(self, item: str):
        """Allow dict-like access for backward compatibility with np.load."""
        return getattr(self, item)

    def __setitem__(self, item: str, value):
        """Allow dict-like assignment for backward compatibility."""
        setattr(self, item, value)

    def to_dict(self):
        """Convert to dictionary for numpy save.

        Handles ragged arrays (ve, gemm_all) by converting to object arrays.
        """
        return {
            "vs": self.vs,
            "edges": self.edges,
            "gemm_edges": self.gemm_edges,
            "gemm_all": (
                np.array(self.gemm_all, dtype=object)
                if self.gemm_all is not None
                else None
            ),
            "sides": self.sides,
            "edges_count": self.edges_count,
            "edge_lengths": self.edge_lengths,
            "edge_areas": self.edge_areas,
            "ve": np.array(self.ve, dtype=object) if self.ve is not None else None,
            "v_mask": self.v_mask,
            "features": self.features,
            "filename": self.filename,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "MeshData":
        """Create MeshData from dictionary (e.g., from np.load).
        
        Handles object arrays (ve, gemm_all) by converting back to lists.
        """
        mesh = cls()
        for key, value in data.items():
            if hasattr(mesh, key):
                # Handle numpy arrays that might be stored as objects
                if key == 'filename':
                    setattr(mesh, key, str(value))
                elif key == 'edges_count':
                    setattr(mesh, key, int(value))
                elif key in ('ve', 'gemm_all') and value is not None:
                    # Convert object arrays back to lists
                    setattr(mesh, key, [list(x) if isinstance(x, np.ndarray) else x for x in value])
                else:
                    setattr(mesh, key, value)
        return mesh

    def validate(self) -> bool:
        """Validate mesh data integrity."""
        if self.vs is None or self.edges is None:
            return False
        if self.edges_count == 0:
            return False
        if self.gemm_edges is None or self.features is None:
            return False
        return True


def fill_mesh(mesh2fill: MeshData, file: str, opt):
    """Fill mesh data from cache or compute from scratch.

    Args:
        mesh2fill: MeshData object to fill
        file: Path to mesh file (.obj)
        opt: Options object with num_aug attribute
    """
    load_path = get_mesh_path(file, opt.num_aug)
    # breakpoint()

    if os.path.exists(load_path):
        # Load from cache
        mesh_data_dict = np.load(load_path, encoding="latin1", allow_pickle=True)
        mesh_data = MeshData.from_dict(mesh_data_dict)
    else:
        # Compute from scratch
        mesh_data = from_scratch(file, opt)

        # Save to cache
        np.savez_compressed(load_path, **mesh_data.to_dict())

    # Copy all attributes to mesh2fill
    for attr in [
        "vs",
        "edges",
        "gemm_edges",
        "gemm_all",
        "edges_count",
        "ve",
        "v_mask",
        "filename",
        "sides",
        "edge_lengths",
        "edge_areas",
        "features",
    ]:
        setattr(mesh2fill, attr, getattr(mesh_data, attr))


def get_mesh_path(file: str, num_aug: int) -> str:
    """Get cache path for mesh preprocessing results.

    Args:
        file: Path to .obj file
        num_aug: Number of augmentation variations

    Returns:
        Path to cached .npz file
    """
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, "cache")
    load_file = os.path.join(
        load_dir, "%s_%03d.npz" % (prefix, np.random.randint(0, num_aug))
    )
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file


def from_scratch(file: str, opt) -> MeshData:
    """Create mesh data from scratch by processing an .obj file.

    Args:
        file: Path to .obj file
        opt: Options object with num_aug attribute

    Returns:
        MeshData object with all computed attributes
    """
    mesh_data = MeshData()
    mesh_data.edge_areas = np.array([], dtype=np.float32)

    # Load vertices and faces from file
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)

    # Remove non-manifold faces
    faces, face_areas = remove_non_manifolds(mesh_data, faces)

    # Apply data augmentation if requested
    if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)

    # Build connectivity (GEMM)
    build_gemm(mesh_data, faces, face_areas)

    # Post-augmentation (vertex sliding)
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)

    # Extract geometric features
    mesh_data.features = extract_features(mesh_data)

    return mesh_data


def fill_from_file(mesh: MeshData, file: str):
    """Load mesh data from .obj file and normalize vertices.

    Args:
        mesh: MeshData object to populate
        file: Path to .obj file

    Returns:
        Tuple of (normalized_vertices, faces)
    """
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == "v":
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == "f":
            face_vertex_ids = [int(c.split("/")[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [
                (ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids
            ]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    # print("vs, faces:", vs)
    # 计算边界框的尺寸
    bbox_min = vs.min(axis=0)
    bbox_max = vs.max(axis=0)
    bbox_sizes = bbox_max - bbox_min

    # 找出最大的边界尺寸
    max_bbox_size = bbox_sizes.max()

    # 将顶点坐标缩放到 [-0.5, 0.5] 范围内
    normalized_vertices = (vs - bbox_min) / max_bbox_size - 0.5

    # 验证归一化顶点坐标
    bbox_min_normalized = normalized_vertices.min(axis=0)
    bbox_max_normalized = normalized_vertices.max(axis=0)

    # print("bbox_min_normalized, bbox_max_normalized:", bbox_min_normalized, bbox_max_normalized)
    # return vs, faces
    return normalized_vertices, faces


def remove_non_manifolds(mesh: MeshData, faces: NDArray[np.int32]):
    """Remove non-manifold faces and compute face areas.

    Args:
        mesh: MeshData object
        faces: Array of faces (#F x 3)

    Returns:
        Tuple of (valid_faces, face_areas)
    """
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
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
    return faces[mask], face_areas[mask]


def build_gemm(
    mesh: MeshData, faces: NDArray[np.int32], face_areas: NDArray[np.float32]
):
    """Build GEMM (Graph Edge Mesh Model) connectivity structure.

    Computes:
    - gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    - sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge
             is in the gemm_edge entry of the 4 neighboring edges
    - gemm_all: all neighboring edges for each edge (variable length)

    Args:
        mesh: MeshData object to populate
        faces: Array of faces (#F x 3)
        face_areas: Area of each face (#F)
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    mesh.edge_areas = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))

                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[
                faces_edges[(idx + 2) % 3]
            ]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = (
                nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            )
            sides[edge_key][nb_count[edge_key] - 1] = (
                nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
            )
    mesh.edges = np.array(edges, dtype=np.int32)
    # print("---------mesh.edges.shape:", mesh.edges.shape)

    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    # print("mesh.gemm_edges.shape:", mesh.gemm_edges.shape)

    all_gemm_edges = []
    for i_gemm in range(mesh.edges.shape[0]):
        e_a, e_b = mesh.edges[i_gemm]
        edge_all_g = []
        ind_a, ind_b = np.where(mesh.edges == e_a)
        for j_e in range(ind_a.shape[0]):
            edge_all_g.append(ind_a[j_e])
        ind_a, ind_b = np.where(mesh.edges == e_b)
        for j_e in range(ind_a.shape[0]):
            edge_all_g.append(ind_a[j_e])
        # print("edge_all_g:", e_a, e_b, len(edge_all_g), edge_all_g)
        edge_all_g = list(set(edge_all_g))
        # edge_all_g.remove(i_gemm)
        # print("unique_list:", e_a, e_b, len(edge_all_g), edge_all_g)
        if len(edge_all_g) < 10:
            # 如果列表长度小于10，补足到10个元素
            edge_all_g.extend([-1] * (10 - len(edge_all_g)))
        elif len(edge_all_g) > 10:
            # 如果列表长度大于10，只保留前10个元素
            edge_all_g = edge_all_g[:10]
        # print("unique_list-1:", e_a, e_b, len(edge_all_g), edge_all_g)
        all_gemm_edges.append(edge_all_g)
    # all_gemm_edges = np.array(all_gemm_edges, dtype=np.int64)
    mesh.gemm_all = np.array(all_gemm_edges, dtype=np.int64)
    # print("all_gemm_edges.shape:", all_gemm_edges.shape, all_gemm_edges)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(
        face_areas
    )  # todo whats the difference between edge_areas and edge_lenghts?


# def compute_face_normals_and_areas(mesh, faces):
#     face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
#                             mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
#     face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
#     face_normals /= face_areas[:, np.newaxis]
#     assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
#     face_areas *= 0.5
#     return face_normals, face_areas
"""法线向量的长度（L2范数）用于计算面积，而后面的规范化操作则是为了将法线向量转换为单位向量"""


def compute_face_normals_and_areas(mesh: MeshData, faces: NDArray[np.int32]):
    """Compute face normals and areas.

    Args:
        mesh: MeshData object
        faces: Array of faces (#F x 3)

    Returns:
        Tuple of (face_normals, face_areas)
    """
    face_normals = np.cross(
        mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
        mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]],
    )
    # Calculate face area from normal vector length (L2 norm)
    face_areas = np.sqrt((face_normals**2).sum(axis=1))
    # print("face_areas.shape:", face_areas.shape, np.sum(face_areas == 0))
    if np.sum(face_areas == 0) > 0:
        # print("mesh.filename:", mesh.filename)
        # print("mesh.vs")
        zero_area_faces_indices = np.where(face_areas == 0)[0]
        # print("zero_area_faces_indices:", zero_area_faces_indices)
        for f in zero_area_faces_indices:
            # print("---mesh.vs[faces[f, 0]], mesh.vs[faces[f, 1]], mesh.vs[faces[f, 2]]:", mesh.vs[faces[f, 0]], mesh.vs[faces[f, 1]], mesh.vs[faces[f, 2]])
            mesh.vs[faces[f, 0]] = mesh.vs[faces[f, 0]] + random.uniform(
                1e-6, 1e-5
            )  # 小幅度调整的大小
            mesh.vs[faces[f, 1]] = mesh.vs[faces[f, 1]] + random.uniform(
                1e-5, 1e-4
            )  # 小幅度调整的大小
            # print("+++mesh.vs[faces[f, 0]], mesh.vs[faces[f, 1]], mesh.vs[faces[f, 2]]:", mesh.vs[faces[f, 0]], mesh.vs[faces[f, 1]], mesh.vs[faces[f, 2]])
        """
        以下代码是对面积为0的三角面重新计算发现和面积
        """
        face_normals = np.cross(
            mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]],
        )
        # 计算每个面的面积。面积是通过计算法线向量的长度（即L2范数）来获得的，这是因为法线向量的长度与面积成正比
        face_areas = np.sqrt((face_normals**2).sum(axis=1))
    # 规范化法线向量意味着将法线向量的长度缩放到1，而不改变其方向。这是通过将法线向量除以其自身的长度（即L2范数）来实现的。
    # 这样做的目的是为了得到一个长度为1的单位法线向量，它仍然保持着原来法线的方向，但长度固定为1
    face_normals /= face_areas[:, np.newaxis]
    assert not np.any(face_areas[:, np.newaxis] == 0), (
        "has zero area face: %s" % mesh.filename
    )
    face_areas *= 0.5
    return face_normals, face_areas


# Data augmentation methods
def augmentation(
    mesh: MeshData, opt, faces: Optional[NDArray[np.int32]] = None
) -> NDArray[np.int32]:
    """Apply data augmentation to mesh.

    Args:
        mesh: MeshData object
        opt: Options with scale_verts and flip_edges attributes
        faces: Array of faces

    Returns:
        Augmented faces array
    """
    if hasattr(opt, "scale_verts") and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, "flip_edges") and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh: MeshData, opt):
    """Apply post-processing augmentation (vertex sliding).

    Args:
        mesh: MeshData object
        opt: Options with slide_verts attribute
    """
    if hasattr(opt, "slide_verts") and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(
        mesh, edge_points
    ).squeeze()  # todo make fixed_division epsilon=0
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (
                    mesh.vs[vi_t] - mesh.vs[vi]
                )
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.15):
    for i in range(mesh.vs.shape[1]):
        # print("---------------------------------mesh.vs.shape, i:", mesh.vs.shape, i)
        # mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)
        mesh.vs[:, i] = mesh.vs[:, i]


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = (
            mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        )
        edge_b = (
            mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        )
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(
                sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]])))
            )
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [
                    [edge_info[1], new_edge[0], new_edge[1]],
                    [edge_info[0], new_edge[0], new_edge[1]],
                ]
            )
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                [
                                    edge_faces[cur_edge_key, 2],
                                    edge_faces[cur_edge_key, 3],
                                ]
                            ):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    # print(flipped)
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face


def check_area(mesh, faces):
    face_normals = np.cross(
        mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
        mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]],
    )
    face_areas = np.sqrt((face_normals**2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(
        mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1
    )
    mesh.edge_lengths = edge_lengths


def mid_point(mesh, features):
    coo_point_x, coo_point_y, coo_point_z = [], [], []
    for edge_id, edge in enumerate(mesh.edges):
        # print("edge_id, edge:", edge_id, edge, list((mesh.vs[edge[0]] + mesh.vs[edge[1]]) / 2))
        coo_point_x.append(((mesh.vs[edge[0]] + mesh.vs[edge[1]]) / 2)[0])
        coo_point_y.append(((mesh.vs[edge[0]] + mesh.vs[edge[1]]) / 2)[1])
        coo_point_z.append(((mesh.vs[edge[0]] + mesh.vs[edge[1]]) / 2)[2])
    # print("len(coo_point):", len(coo_point_x), len(coo_point_y), len(coo_point_z))
    features.append([coo_point_x])
    features.append([coo_point_y])
    features.append([coo_point_z])
    # print("len(features):", len(features))


def extract_features(mesh: MeshData) -> NDArray[np.float32]:
    """Extract geometric features from mesh edges.

    Features include:
    - Dihedral angles
    - Symmetric opposite angles
    - Symmetric ratios (height/base)
    - Edge midpoint coordinates

    Args:
        mesh: MeshData object

    Returns:
        Feature array of shape (5, #E)
    """
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide="raise"):
        try:
            for extractor in [
                dihedral_angle,
                symmetric_opposite_angles,
                symmetric_ratios,
            ]:
                # print("--------------------edge_points:", edge_points)
                feature = extractor(mesh, edge_points)
                # print("mesh, edge_points:", mesh, edge_points, edge_points.shape, mesh.filename)
                # print("extractor, feature:", extractor, len(feature))
                features.append(feature)
            # print("--- np.concatenate(features, axis=0):", mesh.filename, np.concatenate(features, axis=0).shape, mesh.vs.shape, mesh.edges.shape)
            mid_point(mesh, features)
            # print("---  np.concatenate(features, axis=0):", mesh.filename, np.concatenate(features, axis=0).shape)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, "bad features")


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """computes two angles: one for each face shared between the edge
    the angle is in each face opposite the edge
    sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate(
        (np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0
    )
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """computes two ratios: one for each face shared between the edge
    the ratio is between the height / base (edge) of each triangle
    sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate(
        (np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0
    )
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    """returns: edge_points (#E x 4) tensor, with four vertex ids per edge
    for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id
    each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh: MeshData, edge_id: int) -> List[int]:
    """Get the two vertices from adjacent faces for a given edge.

    Args:
        mesh: MeshData object
        edge_id: Edge index

    Returns:
        List of 4 vertex indices [v0, v1, v_adj1, v_adj2]
    """
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [
        edge_a[first_vertex],
        edge_a[1 - first_vertex],
        edge_b[second_vertex],
        edge_d[third_vertex],
    ]


def get_normals(mesh, edge_points, side):
    # print("mesh.vs:", mesh.vs)
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals


def get_opposite_angles(mesh, edge_points, side):
    edges_a = (
        mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    )
    edges_b = (
        mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    )

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[
        :, np.newaxis
    ]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[
        :, np.newaxis
    ]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(
        mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
        ord=2,
        axis=1,
    )
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1
    )
    closest_point = (
        point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    )
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
