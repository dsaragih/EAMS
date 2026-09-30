import torch
from torch_geometric.data import Data
from typing import Optional


class MeshData(Data):
    """
    Custom Data class for mesh data with proper batching behavior.

    This class extends PyTorch Geometric's Data class to handle custom attributes
    that require special index offsetting during batching, including:
    - vertex2face: vertex-to-face mappings
    - face_adj: face adjacency lists
    - fine_to_coarse_{i}: hierarchical vertex mappings

    Required attributes:
    - pos: vertex positions (N, 3)
    - normal: vertex normals (N, 3)
    - edge_index: edge connectivity (2, E)
    - edge_labels: edge labels (E,)
    - face: face connectivity (3, F)
    - face_area: per-face areas (F,)
    """

    def __init__(
        self,
        pos: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_labels: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        face_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pos = pos
        self.normal = normal
        self.edge_index = edge_index
        self.edge_labels = edge_labels
        self.face = face
        self.face_area = face_area
        self.face_labels = face_labels

    def __inc__(self, key: str, value, *args, **kwargs):
        """
        Specify how to increment indices for each attribute during batching.

        PyTorch Geometric calls this method to determine the offset for each attribute
        when combining multiple graphs into a batch.

        Returns:
            int, torch.Tensor, or tuple: The increment value(s) for the given attribute.
        """
        # Handle vertex2face: [vertex_idx, face_idx]
        # Column 0 (vertex) increments by num_nodes, column 1 (face) increments by num_faces
        if key == "vertex2face":
            return torch.tensor([[self.num_nodes, self.num_faces]])

        if key == "vertex2edge":
            return torch.tensor([[self.num_nodes, self.num_edges]])

        # Handle face_adj: [face_idx_1, face_idx_2]
        # Both columns increment by num_faces
        if key == "face_adjacency":
            return torch.tensor([[self.num_faces, self.num_faces]])

        if key == "face_adjacency_edge_index":
            return self.num_edges

        if key == "pointcloud_original_idx":
            return self.num_nodes

        # Handle hierarchical mappings: fine_to_coarse_{i}
        # shape: (num_fine_nodes,) value contains indices to coarse nodes
        # Increment by number of nodes at the coarse level (level i)
        if key.startswith("fine_to_coarse_") and key[-1].isdigit():
            level = int(key.split("_")[-1])
            pos_key = f"pos_{level+1}"
            if hasattr(self, pos_key):
                return getattr(self, pos_key).size(0)
            return 0

        # Handle coarse level face attributes: face_{i}
        # These should increment by number of nodes at level i
        if key.startswith("face_") and key[-1].isdigit():
            level = int(key.split("_")[-1])
            nodes_key = f"pos_{level}"
            if hasattr(self, nodes_key):
                return getattr(self, nodes_key).size(0)

        # Default behavior from parent class
        # This handles standard attributes like edge_index, face, etc.
        return super().__inc__(key, value, *args, **kwargs)

    def from_data(data: Data) -> "MeshData":
        """
        Create a MeshData instance from a generic Data object.

        Args:
            data (Data): A PyTorch Geometric Data object.
        Returns:
            MeshData: A MeshData instance with the same attributes as the input data.
        """
        mesh_data = MeshData()
        for key, item in data:
            mesh_data[key] = item
        return mesh_data

    @property
    def num_faces(self) -> int:
        """Return the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(1)  # face is (3, F) or (K, F)
        return 0
