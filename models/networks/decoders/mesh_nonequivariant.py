import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts import EncoderOutput
from ..encoders.norms import build_norm_layer, canonicalize_norm_type


class MeshNonEquivariantMLP(nn.Module):
    """MLP head with optional graph-aware normalization."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        norm_type: str | None = None,
        use_layernorm: bool = True,
        activation: str = "relu",
        use_bias: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim]

        if norm_type is None:
            norm_type = "layernorm" if use_layernorm else "none"
        if norm_type.lower() == "none":
            self.norm_type = "none"
        else:
            self.norm_type = canonicalize_norm_type(norm_type)
        self.activation = activation

        dims = [input_dim] + list(hidden_dims) + [target_dim]
        self.linears = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=use_bias)
                for i in range(len(dims) - 1)
            ]
        )
        self.norms = nn.ModuleList()
        for i in range(len(dims) - 2):
            if self.norm_type != "none":
                self.norms.append(build_norm_layer(dims[i + 1], self.norm_type))
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(dims) - 2)]
        )

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.2)
        raise ValueError(f"Unsupported activation={self.activation}")

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            if self.norm_type != "none":
                x = self.norms[i](x, batch=batch)
            x = self._act(x)
            x = self.dropouts[i](x)
        return self.linears[-1](x)


class MeshNonEquivariantDecoder(nn.Module):
    """MLP decoder for MeshNonEquivariantEncoder latent states."""

    supports_encoder_frame_averaging = False

    def __init__(
        self,
        node_dim: int | None = None,
        face_dim: int | None = None,
        input_dim: int | None = None,
        edge_attr_dim: int | None = None,
        out_dim: int = 1,
        output_level: str = "vertex",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str | None = None,
        use_layernorm: bool = True,
        use_bias: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        feature_dim = node_dim if node_dim is not None else input_dim
        if output_level not in {"vertex", "edge", "face"}:
            raise ValueError(
                "MeshNonEquivariantDecoder output_level must be one of "
                f"{{'vertex', 'edge', 'face'}}, got: {output_level}"
            )
        if feature_dim is None:
            raise ValueError("MeshNonEquivariantDecoder requires node_dim or input_dim")
        if output_level in {"edge", "face"} and edge_attr_dim is None:
            raise ValueError(
                "MeshNonEquivariantDecoder requires edge_attr_dim for edge/face outputs"
            )
        if output_level == "face" and face_dim is None:
            raise ValueError("MeshNonEquivariantDecoder requires face_dim for face output")

        self.output_level = output_level
        self.eps = eps

        if output_level == "vertex":
            decoder_input_dim = feature_dim + 3 + 3 + 3
        elif output_level == "edge":
            decoder_input_dim = 2 * feature_dim + edge_attr_dim + 3 * 10
        else:
            decoder_input_dim = 3 * feature_dim + 3 * edge_attr_dim + face_dim + 3 * 16

        self.head = MeshNonEquivariantMLP(
            input_dim=decoder_input_dim,
            target_dim=out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            norm_type=norm_type,
            use_layernorm=use_layernorm,
            activation=activation,
            use_bias=use_bias,
        )

    def _require_node_state(self, encoded: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.node_embedding is None:
            raise ValueError("MeshNonEquivariantDecoder requires node_embedding")
        if encoded.latent_pos is None:
            raise ValueError("MeshNonEquivariantDecoder requires latent_pos")
        if encoded.pos.shape != encoded.latent_pos.shape:
            raise ValueError(
                "MeshNonEquivariantDecoder expected pos and latent_pos to have matching "
                f"shapes, got {tuple(encoded.pos.shape)} and {tuple(encoded.latent_pos.shape)}"
            )
        return encoded.node_embedding, encoded.latent_pos

    def _require_edges(self, encoded: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.edge_index is None or encoded.edge_attr is None:
            raise ValueError(
                "MeshNonEquivariantDecoder requires edge_index and edge_attr"
            )
        if encoded.edge_index.ndim != 2 or encoded.edge_index.shape[0] != 2:
            raise ValueError("MeshNonEquivariantDecoder expected edge_index shape (2, E)")
        if encoded.edge_attr.shape[0] != encoded.edge_index.shape[1]:
            raise ValueError(
                "MeshNonEquivariantDecoder expected one edge_attr row per edge"
            )
        return encoded.edge_index.long(), encoded.edge_attr

    def _require_faces(self, encoded: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.face is None:
            raise ValueError("MeshNonEquivariantDecoder requires face connectivity")
        if encoded.face_embedding is None:
            raise ValueError("MeshNonEquivariantDecoder requires face_embedding")
        if encoded.face.ndim != 2 or encoded.face.shape[0] != 3:
            raise ValueError("MeshNonEquivariantDecoder expected face shape (3, F)")
        if encoded.face_embedding.shape[0] != encoded.face.shape[1]:
            raise ValueError(
                "MeshNonEquivariantDecoder expected one face_embedding row per face"
            )
        return encoded.face.long(), encoded.face_embedding

    def _edge_geometry(self, coord: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0].long(), edge_index[1].long()
        delta = coord[src] - coord[dst]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.cat([coord[src], coord[dst], delta, dist], dim=-1)

    def _edge_distance(self, coord: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0].long(), edge_index[1].long()
        delta = coord[src] - coord[dst]
        return delta.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _face_invariant_geometry(self, coord: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        if face.numel() == 0:
            return coord.new_empty((0, 4))

        i, j, k = face[0].long(), face[1].long(), face[2].long()
        r_ij = coord[j] - coord[i]
        r_jk = coord[k] - coord[j]
        r_ik = coord[k] - coord[i]
        normal = torch.cross(r_ij, r_jk, dim=1)
        area = normal.norm(dim=-1, keepdim=True).clamp_min(self.eps) / 2
        return torch.cat(
            [
                r_ij.norm(dim=-1, keepdim=True).clamp_min(self.eps),
                r_jk.norm(dim=-1, keepdim=True).clamp_min(self.eps),
                r_ik.norm(dim=-1, keepdim=True).clamp_min(self.eps),
                area,
            ],
            dim=-1,
        )

    def _face_geometry(self, coord: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        if face.numel() == 0:
            return coord.new_empty((0, 16))

        i, j, k = face[0].long(), face[1].long(), face[2].long()
        r_ij = coord[j] - coord[i]
        r_jk = coord[k] - coord[j]
        normal = torch.cross(r_ij, r_jk, dim=1)
        area = normal.norm(dim=-1, keepdim=True).clamp_min(self.eps) / 2
        centroid = (coord[i] + coord[j] + coord[k]) / 3
        return torch.cat([coord[i], coord[j], coord[k], centroid, normal, area], dim=-1)

    def _lookup_directed_edges(
        self,
        *,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pairs: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if pairs.numel() == 0:
            return edge_attr.new_empty((0, edge_attr.shape[-1]))

        edge_key = edge_index[0].long() * num_nodes + edge_index[1].long()
        order = torch.argsort(edge_key)
        sorted_key = edge_key[order]
        query = pairs[:, 0].long() * num_nodes + pairs[:, 1].long()
        pos = torch.searchsorted(sorted_key, query)
        safe_pos = pos.clamp_max(max(sorted_key.numel() - 1, 0))
        found = (pos < sorted_key.numel()) & (sorted_key[safe_pos] == query)
        if not bool(found.all()):
            missing = int((~found).sum().item())
            raise ValueError(
                "MeshNonEquivariantDecoder could not find directed face edges in "
                f"edge_index; missing {missing} directed edge(s)"
            )
        return edge_attr[order[safe_pos]]

    def forward_vertex(self, encoded: EncoderOutput) -> torch.Tensor:
        z, u = self._require_node_state(encoded)
        d = u - encoded.pos
        features = torch.cat([z, encoded.pos, u, d], dim=-1)
        return self.head(features, encoded.batch)

    def forward_edge(self, encoded: EncoderOutput) -> torch.Tensor:
        z, u = self._require_node_state(encoded)
        edge_index, edge_attr = self._require_edges(encoded)
        d = u - encoded.pos
        features = torch.cat(
            [
                z[edge_index[0].long()],
                z[edge_index[1].long()],
                edge_attr,
                self._edge_geometry(encoded.pos, edge_index),
                self._edge_geometry(u, edge_index),
                self._edge_geometry(d, edge_index),
            ],
            dim=-1,
        )
        return self.head(features, encoded.batch[edge_index[0].long()])

    def forward_face(self, encoded: EncoderOutput) -> torch.Tensor:
        z, u = self._require_node_state(encoded)
        edge_index, edge_attr = self._require_edges(encoded)
        face, q = self._require_faces(encoded)
        d = u - encoded.pos

        i, j, k = face[0].long(), face[1].long(), face[2].long()
        face_pairs = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([j, k], dim=-1),
                torch.stack([i, k], dim=-1),
            ],
            dim=1,
        )
        face_edges = self._lookup_directed_edges(
            edge_index=edge_index,
            edge_attr=edge_attr,
            pairs=face_pairs.reshape(-1, 2),
            num_nodes=encoded.pos.shape[0],
        ).reshape(face.shape[1], 3, edge_attr.shape[-1])

        features = torch.cat(
            [
                z[i],
                z[j],
                z[k],
                face_edges[:, 0],
                face_edges[:, 1],
                face_edges[:, 2],
                q,
                self._face_geometry(encoded.pos, face),
                self._face_geometry(u, face),
                self._face_geometry(d, face),
            ],
            dim=-1,
        )
        face_batch = encoded.face_batch
        if face_batch is None:
            face_batch = (
                encoded.batch[i]
                if face.numel() > 0
                else encoded.batch.new_empty((0,), dtype=encoded.batch.dtype)
            )
        return self.head(features, face_batch)

    def forward(self, encoded: EncoderOutput) -> dict[str, torch.Tensor | str]:
        if self.output_level == "vertex":
            pred = self.forward_vertex(encoded)
        elif self.output_level == "edge":
            pred = self.forward_edge(encoded)
        else:
            pred = self.forward_face(encoded)
        return {"pred": pred, "level": self.output_level}


class MeshNonEquivariantLatentCoordDecoder(MeshNonEquivariantDecoder):
    """Frame-averaging-compatible decoder that only uses collapsed latent coordinates."""

    supports_encoder_frame_averaging = True

    def __init__(
        self,
        node_dim: int | None = None,
        face_dim: int | None = None,
        input_dim: int | None = None,
        edge_attr_dim: int | None = None,
        out_dim: int = 1,
        output_level: str = "vertex",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str | None = None,
        use_layernorm: bool = True,
        use_bias: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        nn.Module.__init__(self)
        del kwargs

        feature_dim = node_dim if node_dim is not None else input_dim
        if output_level not in {"vertex", "edge", "face"}:
            raise ValueError(
                "MeshNonEquivariantLatentCoordDecoder output_level must be one of "
                f"{{'vertex', 'edge', 'face'}}, got: {output_level}"
            )
        if feature_dim is None:
            raise ValueError(
                "MeshNonEquivariantLatentCoordDecoder requires node_dim or input_dim"
            )
        if output_level in {"edge", "face"} and edge_attr_dim is None:
            raise ValueError(
                "MeshNonEquivariantLatentCoordDecoder requires edge_attr_dim for edge/face outputs"
            )
        if output_level == "face" and face_dim is None:
            raise ValueError(
                "MeshNonEquivariantLatentCoordDecoder requires face_dim for face output"
            )

        self.output_level = output_level
        self.eps = eps

        if output_level == "vertex":
            decoder_input_dim = feature_dim + 1
        elif output_level == "edge":
            decoder_input_dim = 2 * feature_dim + edge_attr_dim + 1
        else:
            decoder_input_dim = 3 * feature_dim + 3 * edge_attr_dim + face_dim + 4

        self.head = MeshNonEquivariantMLP(
            input_dim=decoder_input_dim,
            target_dim=out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            norm_type=norm_type,
            use_layernorm=use_layernorm,
            activation=activation,
            use_bias=use_bias,
        )

    def forward_vertex(self, encoded: EncoderOutput) -> torch.Tensor:
        z, u = self._require_node_state(encoded)
        features = torch.cat([z, u.norm(dim=-1, keepdim=True)], dim=-1)
        return self.head(features, encoded.batch)

    def forward_edge(self, encoded: EncoderOutput) -> torch.Tensor:
        z, u = self._require_node_state(encoded)
        edge_index, edge_attr = self._require_edges(encoded)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        features = torch.cat(
            [
                z[src],
                z[dst],
                edge_attr,
                self._edge_distance(u, edge_index),
            ],
            dim=-1,
        )
        return self.head(features, encoded.batch[src])

    def forward_face(self, encoded: EncoderOutput) -> torch.Tensor:
        z, u = self._require_node_state(encoded)
        edge_index, edge_attr = self._require_edges(encoded)
        face, q = self._require_faces(encoded)

        i, j, k = face[0].long(), face[1].long(), face[2].long()
        face_pairs = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([j, k], dim=-1),
                torch.stack([i, k], dim=-1),
            ],
            dim=1,
        )
        face_edges = self._lookup_directed_edges(
            edge_index=edge_index,
            edge_attr=edge_attr,
            pairs=face_pairs.reshape(-1, 2),
            num_nodes=encoded.pos.shape[0],
        ).reshape(face.shape[1], 3, edge_attr.shape[-1])

        features = torch.cat(
            [
                z[i],
                z[j],
                z[k],
                face_edges[:, 0],
                face_edges[:, 1],
                face_edges[:, 2],
                q,
                self._face_invariant_geometry(u, face),
            ],
            dim=-1,
        )
        face_batch = encoded.face_batch
        if face_batch is None:
            face_batch = (
                encoded.batch[i]
                if face.numel() > 0
                else encoded.batch.new_empty((0,), dtype=encoded.batch.dtype)
            )
        return self.head(features, face_batch)
