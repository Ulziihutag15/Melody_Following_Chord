import math

import torch
from torch import nn

from .encoder import SinusoidalPositionalEncoding


class MelodyDecoder(nn.Module):
    """Transformer decoder for autoregressive melody prediction.

    Args:
        melody_vocab_size: Number of melody tokens, including PAD/START/REST.
        d_model: Shared hidden size used by encoder and decoder.
        num_layers: Number of Transformer decoder layers.
        num_heads: Number of attention heads.
        dim_feedforward: Size of the feed-forward layer inside each block.
        dropout: Dropout probability.
        pad_id: Padding token ID. In the current preprocessing, PAD is 0.
        max_len: Maximum melody sequence length supported by positional encoding.

    Inputs:
        melody_input_ids: LongTensor shaped [batch, melody_time]
        chord_memory: FloatTensor shaped [batch, chord_time, d_model]
        chord_padding_mask: BoolTensor shaped [batch, chord_time], True where padded

    Output:
        logits: FloatTensor shaped [batch, melody_time, melody_vocab_size]
    """

    def __init__(
        self,
        melody_vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 1024,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(
            num_embeddings=melody_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )
        self.position = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, melody_vocab_size)

    def forward(
        self,
        melody_input_ids: torch.Tensor,
        chord_memory: torch.Tensor,
        chord_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        melody_padding_mask = melody_input_ids.eq(self.pad_id)
        causal_mask = self._causal_mask(
            melody_input_ids.size(1), device=melody_input_ids.device
        )

        x = self.embedding(melody_input_ids) * math.sqrt(self.d_model)
        x = self.position(x)

        hidden = self.decoder(
            tgt=x,
            memory=chord_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=melody_padding_mask,
            memory_key_padding_mask=chord_padding_mask,
        )
        hidden = self.norm(hidden)
        return self.output(hidden)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((size, size), dtype=torch.bool, device=device),
            diagonal=1,
        )
