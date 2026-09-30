import torch
import torch.nn as nn

from loguru import logger

from ..contracts import EncoderOutput
from ..mlp_decoder import MLP_Decoder
from ..mlp_decoder import EdgePositionMessageDecoder
from .utils import resolve_face_features

ARITHMETIC_REDUCTION = ["mean", "max", "sum"]


class EdgePositionDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int | None = None,
        face_dim: int | None = None,
        input_dim: int | None = None,
        edge_attr_dim: int | None = None,
        out_dim: int = 1,
        output_level: str = "edge",
        edge_aggregate: str = "sum",
        face_source: str = "auto",
        face_reduce: str = "mean",
        dist_emb_dim: int = 16,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str | None = None,
        use_layernorm: bool = True,
        msg_hidden_dims: list[int] | None = None,
        dist_hidden_dims: list[int] | None = None,
        face_hidden_dims: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()

        feature_dim = node_dim if node_dim is not None else input_dim
        if output_level not in {"edge", "vertex", "face"}:
            raise ValueError(
                f"output_level must be one of {{'edge','vertex','face'}}, got: {output_level}"
            )

        self.output_level = output_level
        self.edge_aggregate = edge_aggregate
        self.face_source = face_source
        self.face_reduce = face_reduce

        if output_level == "face":
            if face_source == "face_embedding":
                face_feature_dim = face_dim
            elif face_source == "auto":
                face_feature_dim = face_dim if face_dim is not None else feature_dim
            else:
                face_feature_dim = feature_dim
            if face_feature_dim is None:
                raise ValueError(
                    "Face decoding requires face_dim or node_dim/input_dim"
                )
            self.face_head = MLP_Decoder(
                input_dim=face_feature_dim,
                target_dim=out_dim,
                hidden_dims=(
                    face_hidden_dims
                    if face_hidden_dims is not None
                    else msg_hidden_dims
                ),
                dropout=dropout,
                norm_type=norm_type,
                use_layernorm=use_layernorm,
                activation=activation,
            )
            self.message_decoder = None
        else:
            if feature_dim is None:
                raise ValueError("EdgePositionDecoder requires node_dim or input_dim")
            if edge_attr_dim is None:
                raise ValueError("EdgePositionDecoder requires edge_attr_dim")
            self.message_decoder = EdgePositionMessageDecoder(
                node_dim=feature_dim,
                edge_attr_dim=edge_attr_dim,
                out_dim=out_dim,
                msg_hidden_dims=msg_hidden_dims,
                dist_hidden_dims=dist_hidden_dims,
                dist_emb_dim=dist_emb_dim,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type,
                use_layernorm=use_layernorm,
            )
            self.face_head = None

    def _require_edges(self, encoded: EncoderOutput):
        if encoded.edge_index is None or encoded.edge_attr is None:
            raise ValueError(
                "EdgePositionDecoder requires encoder outputs with edge_index and edge_attr"
            )

    def forward_edge(self, encoded: EncoderOutput) -> torch.Tensor:
        self._require_edges(encoded)
        pred, _, _ = self.message_decoder.forward_edge(
            x=encoded.node_embedding,
            pos=encoded.pos,
            edge_index=encoded.edge_index,
            edge_attr=encoded.edge_attr,
            batch=encoded.batch,
        )
        return pred

    def forward_vertex(self, encoded: EncoderOutput) -> torch.Tensor:
        self._require_edges(encoded)
        reduce = (
            self.edge_aggregate
            if self.edge_aggregate in ARITHMETIC_REDUCTION
            else "sum"
        )
        if self.edge_aggregate not in ARITHMETIC_REDUCTION:
            logger.warning(
                f"output_level='vertex' expects edge_aggregate in {ARITHMETIC_REDUCTION}; "
                f"got edge_aggregate={self.edge_aggregate}. Falling back to reduce='sum'."
            )
        return self.message_decoder.forward_vertex(
            x=encoded.node_embedding,
            pos=encoded.pos,
            edge_index=encoded.edge_index,
            edge_attr=encoded.edge_attr,
            batch=encoded.batch,
            reduce=reduce,
        )

    def forward_face(self, encoded: EncoderOutput) -> torch.Tensor:
        if self.face_head is None:
            raise ValueError("Face decoding requested but face_head is not configured")
        face_features, face_batch = resolve_face_features(
            encoded,
            face_source=self.face_source,
            face_reduce=self.face_reduce,
        )
        return self.face_head(face_features, face_batch)

    def forward(self, encoded: EncoderOutput) -> dict[str, torch.Tensor | str]:
        if self.output_level == "edge":
            pred = self.forward_edge(encoded)
        elif self.output_level == "face":
            pred = self.forward_face(encoded)
        else:
            pred = self.forward_vertex(encoded)
        return {"pred": pred, "level": self.output_level}
