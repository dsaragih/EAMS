import torch
import torch.nn as nn

from ..contracts import EncoderOutput
from ..mlp_decoder import MLP_Decoder
from .utils import resolve_face_features


class MLPVertexDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int | None = None,
        face_dim: int | None = None,
        input_dim: int | None = None,
        out_dim: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        norm_type: str | None = None,
        use_layernorm: bool = True,
        activation: str = "relu",
        use_bias: bool = True,
        output_level: str = "vertex",
        face_source: str = "auto",
        face_reduce: str = "mean",
        **kwargs,
    ):
        super().__init__()

        feature_dim = node_dim if node_dim is not None else input_dim
        if output_level not in {"vertex", "face"}:
            raise ValueError(
                f"MLPVertexDecoder only supports output_level in {{'vertex', 'face'}}, got: {output_level}"
            )
        if output_level == "vertex" and feature_dim is None:
            raise ValueError("MLPVertexDecoder requires node_dim or input_dim")
        if output_level == "face":
            if face_source == "face_embedding":
                feature_dim = face_dim
            elif face_source == "auto":
                feature_dim = face_dim if face_dim is not None else feature_dim
            if feature_dim is None:
                raise ValueError(
                    "Face MLP decoding requires face_dim or node_dim/input_dim"
                )

        self.output_level = output_level
        self.face_source = face_source
        self.face_reduce = face_reduce
        self.head = MLP_Decoder(
            input_dim=feature_dim,
            target_dim=out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            norm_type=norm_type,
            use_layernorm=use_layernorm,
            activation=activation,
            use_bias=use_bias,
        )

    def forward(self, encoded: EncoderOutput) -> dict[str, torch.Tensor | str]:
        if self.output_level == "vertex":
            if encoded.node_embedding is None:
                raise ValueError("Vertex decoding requires encoder node_embedding")
            pred = self.head(encoded.node_embedding, encoded.batch)
        else:
            face_features, face_batch = resolve_face_features(
                encoded,
                face_source=self.face_source,
                face_reduce=self.face_reduce,
            )
            pred = self.head(face_features, face_batch)
        return {"pred": pred, "level": self.output_level}
