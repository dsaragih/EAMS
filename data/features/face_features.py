import trimesh
from data.types import MeshData
from torch_geometric.data import Batch
from typing import Union
import torch

import numpy as np




def compute_di_angle(data: Union[MeshData, Batch]):
    # Handle batch processing
    if isinstance(data, Batch):
        dihedral_angles = []
        graphs = data.to_data_list()

        for graph in graphs:
            dihedral_angle = _compute_di_angle_single(graph)
            dihedral_angles.append(dihedral_angle)

        return torch.cat(dihedral_angles, dim=0)

    # Handle single graph
    return _compute_di_angle_single(data)


def _compute_di_angle_single(data: MeshData):
    """Helper function to compute dihedral angles for a single graph."""
    mesh = trimesh.Trimesh(
        vertices=data.pos.cpu(), faces=data.face.T.cpu(), process=False
    )

    dihedral_angle = list()
    for i in range(mesh.faces.shape[0]):
        dihedral_angle.append(list())

    face_adjacency = mesh.face_adjacency

    for adj_faces in face_adjacency:
        dihedral_angle[adj_faces[0]].append(
            np.abs(
                np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])
            )
        )
        dihedral_angle[adj_faces[1]].append(
            np.abs(
                np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])
            )
        )

    for i, l in enumerate(dihedral_angle):
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) != 3:
            print(i, "Padding Failed")
    face_dihedral_angle = np.array(dihedral_angle).reshape(-1, 3)

    return torch.tensor(face_dihedral_angle).to(data.pos.device)
