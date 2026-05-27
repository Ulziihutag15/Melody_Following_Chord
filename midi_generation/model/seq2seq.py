from torch import nn

from .decoder import MelodyDecoder
from .encoder import ChordEncoder


class ChordConditionedMelodyModel(nn.Module):
    """Encoder-decoder model for chord-conditioned melody generation."""

    def __init__(
        self,
        chord_vocab_size: int,
        melody_vocab_size: int,
        d_model: int = 128,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 1024,
        use_chord_features: bool = False,
        chord_feature_matrix=None,
    ):
        super().__init__()
        self.encoder = ChordEncoder(
            chord_vocab_size=chord_vocab_size,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
            use_chord_features=use_chord_features,
            chord_feature_matrix=chord_feature_matrix,
        )
        self.decoder = MelodyDecoder(
            melody_vocab_size=melody_vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
        )

    def forward(self, chord_ids, melody_input_ids):
        chord_memory, chord_padding_mask = self.encoder(chord_ids)
        return self.decoder(
            melody_input_ids=melody_input_ids,
            chord_memory=chord_memory,
            chord_padding_mask=chord_padding_mask,
        )
