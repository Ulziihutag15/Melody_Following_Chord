import torch


ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

QUALITY_INTERVALS = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
    "+": [0, 4, 8],
    "sus4": [0, 5, 7],
    "m7b5": [0, 3, 6, 10],
}


def parse_chord(chord: str) -> tuple[str, str]:
    if len(chord) >= 2 and chord[1] in {"#", "b"}:
        return chord[:2], chord[2:]
    return chord[:1], chord[1:]


def chord_pitch_class_vector(chord: str) -> list[float]:
    vector = [0.0] * 12
    root, quality = parse_chord(chord)
    root_pc = ROOT_TO_PC.get(root)
    intervals = QUALITY_INTERVALS.get(quality)
    if root_pc is None or intervals is None:
        return vector

    for interval in intervals:
        vector[(root_pc + interval) % 12] = 1.0
    return vector


def build_chord_feature_matrix(chord_vocab: dict[str, int]) -> torch.Tensor:
    matrix = torch.zeros(len(chord_vocab), 12, dtype=torch.float32)
    for chord, chord_id in chord_vocab.items():
        matrix[chord_id] = torch.tensor(chord_pitch_class_vector(chord))
    return matrix
