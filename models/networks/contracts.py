from dataclasses import dataclass

import torch


@dataclass
class EncoderOutput:
    node_embedding: torch.Tensor | None
    pos: torch.Tensor
    batch: torch.Tensor
    latent_pos: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    face: torch.Tensor | None = None
    face_batch: torch.Tensor | None = None
    face_embedding: torch.Tensor | None = None
    assignment_matrix: torch.Tensor | None = None
    region_features: torch.Tensor | None = None
    region_coords: torch.Tensor | None = None
    region_mass: torch.Tensor | None = None
    virtual_node_features: torch.Tensor | None = None
    virtual_node_coords: torch.Tensor | None = None
