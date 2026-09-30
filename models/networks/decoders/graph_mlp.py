import torch
import torch.nn as nn

from ..contracts import EncoderOutput
from ..mlp_decoder import MLP_Decoder


class GraphMLPDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int | None = None,
        input_dim: int | None = None,
        out_dim: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        norm_type: str | None = None,
        use_layernorm: bool = True,
        activation: str = "relu",
        use_bias: bool = True,
        output_level: str = "graph",
        apply_log_softmax: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        feature_dim = node_dim if node_dim is not None else input_dim
        if feature_dim is None:
            raise ValueError("GraphMLPDecoder requires node_dim or input_dim")
        if output_level != "graph":
            raise ValueError(
                f"GraphMLPDecoder only supports output_level='graph', got: {output_level}"
            )

        self.output_level = output_level
        self.apply_log_softmax = apply_log_softmax
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
        if encoded.node_embedding is None:
            raise ValueError("Graph decoding requires encoder node_embedding")
        pred = self.head(encoded.node_embedding, encoded.batch)
        if self.apply_log_softmax:
            pred = torch.log_softmax(pred, dim=-1)
        return {"pred": pred, "level": self.output_level}
