import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed positional information to token embeddings. 
    Need this so that same chords following different sequences
    map to different contexts."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ChordEncoder(nn.Module):
    """Transformer encoder for chord progressions.

    Args:
        chord_vocab_size: Number of chord tokens, including PAD/UNK.
        d_model: Shared hidden size used by encoder and decoder.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dim_feedforward: Size of the feed-forward layer inside each block.
        dropout: Dropout probability.
        pad_id: Padding token ID. In the current preprocessing, PAD is 0.
        max_len: Maximum chord sequence length supported by positional encoding.

    Input:
        chord_ids: LongTensor shaped [batch, chord_time]

    Output:
        memory: FloatTensor shaped [batch, chord_time, d_model]
        padding_mask: BoolTensor shaped [batch, chord_time], True where padded
    """

    def __init__(
        self,
        chord_vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 1024,
        use_chord_features: bool = False,
        chord_feature_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.use_chord_features = use_chord_features

        self.embedding = nn.Embedding(
            num_embeddings=chord_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )
        if use_chord_features:
            if chord_feature_matrix is None:
                chord_feature_matrix = torch.zeros(chord_vocab_size, 12)
            self.register_buffer("chord_feature_matrix", chord_feature_matrix.float())
            self.chord_feature_proj = nn.Linear(12, d_model, bias=False)
        self.position = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, chord_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = chord_ids.eq(self.pad_id)

        x = self.embedding(chord_ids) * math.sqrt(self.d_model)
        if self.use_chord_features:
            chord_features = self.chord_feature_matrix[chord_ids]
            x = x + self.chord_feature_proj(chord_features)
        x = self.position(x)
        memory = self.encoder(x, src_key_padding_mask=padding_mask)
        memory = self.norm(memory)

        return memory, padding_mask
