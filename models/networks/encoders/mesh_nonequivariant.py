import torch
import torch.nn as nn

from data.types import MeshData
from models.layers.mesh_nonequivariant_conv import MeshNonEquivariantConv
from models.networks.encoders.norms import build_norm_layer, canonicalize_norm_type

from ..contracts import EncoderOutput


class MeshNonEquivariantEncoder(nn.Module):
    """Non-equivariant mesh encoder with persistent node, edge, face, and u states."""

    def __init__(
        self,
        input_dim: int,
        edges_input_dim: int,
        width: int,
        width_mult: list[int],
        edge_state_dim: int = 32,
        face_state_dim: int = 32,
        face_input_dim: int = 0,
        layer_hidden_dim: int | None = None,
        residual: bool = True,
        pre_norm: bool = True,
        norm_type: str = "stateless_batchnorm",
        detect_nan: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got: {input_dim}")
        if edges_input_dim < 1:
            raise ValueError(
                "MeshNonEquivariantEncoder requires configured edge features "
                f"(edges_input_dim > 0), got: {edges_input_dim}"
            )
        if edge_state_dim < 1:
            raise ValueError(f"edge_state_dim must be positive, got: {edge_state_dim}")
        if face_state_dim < 1:
            raise ValueError(f"face_state_dim must be positive, got: {face_state_dim}")
        if len(width_mult) < 2:
            raise ValueError("width_mult must contain at least input and output widths")

        self.input_dim = input_dim
        self.base_edges_input_dim = edges_input_dim
        self.edge_state_dim = edge_state_dim
        self.face_state_dim = face_state_dim
        self.face_input_dim = face_input_dim
        self.width = width
        self.width_mult = width_mult
        self.num_layers = len(width_mult) - 1
        self.residual = residual
        self.pre_norm = pre_norm
        self.detect_nan = detect_nan
        self.norm_type = canonicalize_norm_type(norm_type)

        self.input_norm = build_norm_layer(input_dim, self.norm_type)
        self.embed_norm = build_norm_layer(width, self.norm_type)
        self.node_in = nn.Linear(input_dim, width)
        self.u_in = nn.Linear(6, 3)
        self.edge_in = nn.Linear(edges_input_dim, edge_state_dim)

        face_init_input_dim = face_input_dim + 1 + 3 + 3 + 3 * width
        self.face_in = nn.Sequential(
            nn.Linear(face_init_input_dim, max(width, face_state_dim)),
            nn.SiLU(),
            nn.Linear(max(width, face_state_dim), face_state_dim),
            nn.SiLU(),
        )

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = width * width_mult[i]
            out_dim = width * width_mult[i + 1]
            self.conv_layers.append(
                MeshNonEquivariantConv(
                    input_nf=in_dim,
                    output_nf=out_dim,
                    edge_nf=edge_state_dim,
                    face_nf=face_state_dim,
                    hidden_nf=layer_hidden_dim,
                    residual=residual,
                )
            )

        self.output_dim = width * width_mult[-1]
        self.face_output_dim = face_state_dim
        # BaseModule passes this to edge decoders, so expose the learned edge state dim.
        self.edges_input_dim = edge_state_dim

    @property
    def required_batch_attributes(self) -> list[str]:
        return ["edge_index", "edge_attr", "pos", "normal", "batch", "x", "face"]

    def _check_finite(
        self, tensor: torch.Tensor | None, name: str, error_message: str
    ) -> None:
        if tensor is None or not self.detect_nan:
            return
        if not torch.isfinite(tensor).all():
            raise ValueError(error_message or f"Non-finite values in {name}")

    def _prepare_node_state(self, data: MeshData) -> torch.Tensor:
        z = data.x.float()
        batch = data.batch.long()
        self._check_finite(z, "input node features", "Non-finite node features")
        if self.pre_norm:
            z = self.input_norm(z, batch=batch)
        z = self.node_in(z)
        if not self.pre_norm:
            z = self.embed_norm(z, batch=batch)
        z = torch.nn.functional.silu(z)
        return z

    def _prepare_edge_state(self, data: MeshData) -> torch.Tensor:
        edge_attr = data.edge_attr.float()
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.shape[0] != data.edge_index.shape[1]:
            raise ValueError(
                "Expected data.edge_attr to have one row per edge, got: "
                f"{edge_attr.shape[0]} rows for {data.edge_index.shape[1]} edges"
            )
        if edge_attr.shape[-1] != self.base_edges_input_dim:
            raise ValueError(
                "Expected data.edge_attr last dimension to match configured "
                f"edges_input_dim={self.base_edges_input_dim}, got: {edge_attr.shape[-1]}"
            )
        self._check_finite(edge_attr, "edge features", "Non-finite edge features")
        return torch.nn.functional.silu(self.edge_in(edge_attr))

    def _prepare_face_attr(self, data: MeshData, face: torch.Tensor) -> torch.Tensor:
        num_faces = face.shape[1]
        face_attr = getattr(data, "face_attr", None)
        if face_attr is None:
            return face.new_zeros((num_faces, self.face_input_dim), dtype=torch.float32)
        face_attr = face_attr.float()
        if face_attr.ndim == 1:
            face_attr = face_attr.unsqueeze(-1)
        if face_attr.shape[0] != num_faces:
            raise ValueError(
                f"Expected data.face_attr to have one row per face ({num_faces}), "
                f"got: {face_attr.shape[0]}"
            )
        if face_attr.shape[-1] != self.face_input_dim:
            raise ValueError(
                "Expected data.face_attr last dimension to match configured "
                f"face_input_dim={self.face_input_dim}, got: {face_attr.shape[-1]}"
            )
        self._check_finite(face_attr, "face features", "Non-finite face features")
        return face_attr

    def _face_geometry(
        self, pos: torch.Tensor, face: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j, k = face[0].long(), face[1].long(), face[2].long()
        r_ij = pos[j] - pos[i]
        r_jk = pos[k] - pos[j]
        normal = torch.cross(r_ij, r_jk, dim=1)
        area = normal.norm(dim=-1, keepdim=True) / 2
        centroid = (pos[i] + pos[j] + pos[k]) / 3
        return area, normal, centroid

    def _prepare_face_state(
        self,
        *,
        data: MeshData,
        z: torch.Tensor,
        pos: torch.Tensor,
        face: torch.Tensor,
    ) -> torch.Tensor:
        if face.numel() == 0:
            return z.new_empty((0, self.face_state_dim))

        face_attr = self._prepare_face_attr(data, face).to(device=z.device, dtype=z.dtype)
        area, normal, centroid = self._face_geometry(pos, face)
        i, j, k = face[0].long(), face[1].long(), face[2].long()
        face_input = torch.cat(
            [face_attr, area, normal, centroid, z[i], z[j], z[k]], dim=-1
        )
        return self.face_in(face_input)

    def _validate_inputs(self, data: MeshData) -> None:
        for attr in self.required_batch_attributes:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                raise ValueError(f"Input data is missing required attribute '{attr}'")

    def forward(self, data: MeshData) -> EncoderOutput:
        self._validate_inputs(data)

        pos = data.pos.float()
        normal = data.normal.float()
        batch = data.batch.long()
        edge_index = data.edge_index.long()
        face = data.face.long()

        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"Expected data.pos with shape (N, 3), got: {tuple(pos.shape)}")
        if normal.shape != pos.shape:
            raise ValueError(
                f"Expected data.normal with shape {tuple(pos.shape)}, got: {tuple(normal.shape)}"
            )
        if batch.ndim != 1 or batch.shape[0] != pos.shape[0]:
            raise ValueError("Expected data.batch to align with data.pos")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Expected data.edge_index with shape (2, E)")
        if face.ndim != 2 or face.shape[0] != 3:
            raise ValueError("Expected data.face with shape (3, F)")

        self._check_finite(pos, "input positions", "Non-finite positions")
        self._check_finite(normal, "input normals", "Non-finite normals")

        z = self._prepare_node_state(data)
        u = self.u_in(torch.cat([pos, normal], dim=-1))
        edge_state = self._prepare_edge_state(data)
        face_state = self._prepare_face_state(data=data, z=z, pos=pos, face=face)

        if self.detect_nan:
            for tensor, name in (
                (z, "node embedding"),
                (u, "latent positions"),
                (edge_state, "edge state"),
                (face_state, "face state"),
            ):
                # breakpoint()
                self._check_finite(tensor, name, f"Non-finite values in {name}")

        for conv_layer in self.conv_layers:
            z, u, edge_state, face_state = conv_layer(
                z=z,
                x=pos,
                u=u,
                edge_index=edge_index,
                edge_state=edge_state,
                face=face,
                face_state=face_state,
            )

        if self.detect_nan:
            for tensor, name in (
                (z, "node embedding"),
                (u, "latent positions"),
                (edge_state, "edge state"),
                (face_state, "face state"),
            ):
                self._check_finite(tensor, name, f"Non-finite values in {name}")

        face_batch = (
            batch[face[0].long()]
            if face.numel() > 0
            else batch.new_empty((0,), dtype=batch.dtype)
        )
        return EncoderOutput(
            node_embedding=z,
            pos=pos,
            batch=batch,
            latent_pos=u,
            edge_index=edge_index,
            edge_attr=edge_state,
            face=face,
            face_batch=face_batch,
            face_embedding=face_state,
        )
