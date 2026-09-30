"""Compatibility module for EMNN-related imports.

The EMNN method now lives under models.networks.encoders.emnn, while the
generic encoder-decoder composition wrapper lives in models.networks.encoder_decoder.
"""

from models.networks.contracts import EncoderOutput
from models.networks.encoder_decoder import EncoderDecoderNetwork
from models.networks.encoders.emnn import EMNN, log_stats

__all__ = ["EMNN", "EncoderDecoderNetwork", "EncoderOutput", "log_stats"]
