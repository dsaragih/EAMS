import torch
import torch.nn as nn

from data.types import MeshData
from loguru import logger


class EncoderDecoderNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        detect_nan: bool | None = None,
        output_embedding_key: str = "node_embedding",
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.detect_nan = detect_nan
        self.output_embedding_key = output_embedding_key

    def extract_embedding(self, encoded) -> torch.Tensor:
        if hasattr(encoded, self.output_embedding_key):
            value = getattr(encoded, self.output_embedding_key)
            if value is not None:
                return value
        if isinstance(encoded, dict) and self.output_embedding_key in encoded:
            value = encoded[self.output_embedding_key]
            if value is not None:
                return value
        raise ValueError(
            f"Encoder output is missing embedding key '{self.output_embedding_key}'"
        )

    def encode(self, data: MeshData):
        return self.encoder(data)

    def decode(self, encoded):
        return self.decoder(encoded)

    def package_output(self, *, encoded, decoded):
        decoded_extras = {}

        if torch.is_tensor(decoded):
            pred = decoded
            level = getattr(self.decoder, "output_level", None)
        elif isinstance(decoded, dict):
            if "pred" not in decoded:
                raise ValueError("Decoder output dict must include 'pred'")
            pred = decoded["pred"]
            level = decoded.get("level", getattr(self.decoder, "output_level", None))
            decoded_extras = {
                key: value for key, value in decoded.items() if key not in {"pred", "level"}
            }
        else:
            raise ValueError(
                f"Decoder output must be a tensor or dict, got: {type(decoded)}"
            )

        embedding = self.extract_embedding(encoded)
        detect_nan = self.detect_nan
        if detect_nan is None:
            detect_nan = getattr(self.encoder, "detect_nan", False)

        if detect_nan and torch.isnan(pred).any():
            logger.error("NaN detected in model output!")
            logger.error(f"  Output shape: {pred.shape}")
            logger.error(
                f"  NaN count: {torch.isnan(pred).sum().item()}/{pred.numel()}"
            )
            logger.error(
                f"  Encoder features - min: {embedding.min().item():.4f}, max: {embedding.max().item():.4f}, has_nan: {torch.isnan(embedding).any()}"
            )
            raise ValueError("NaN detected in model output")

        output = {"pred": pred, "embedding": embedding, "level": level}
        output.update(decoded_extras)
        for key in (
            "assignment_matrix",
            "region_features",
            "region_coords",
            "region_mass",
            "latent_pos",
            "virtual_node_features",
            "virtual_node_coords",
        ):
            if hasattr(encoded, key):
                value = getattr(encoded, key)
                if value is not None:
                    output[key] = value

        return output

    def forward(self, data: MeshData):
        encoded = self.encode(data)
        decoded = self.decode(encoded)
        return self.package_output(encoded=encoded, decoded=decoded)
