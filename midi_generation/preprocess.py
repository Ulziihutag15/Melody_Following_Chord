#!/usr/bin/env python3
"""
Preprocess COMMU chord progressions and MIDI melodies.

Outputs:
  preprocessed/chord_vocab.json
  preprocessed/melody_vocab.json
  preprocessed/train.jsonl
  preprocessed/val.jsonl

Each JSONL example contains time-aligned chord IDs and melody IDs. Variable
lengths are preserved so a training Dataset can pad batches later.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PAD = "<PAD>"
UNK = "<UNK>"
START = "<START>"
REST = "REST"
HOLD = "HOLD"


def parse_chord_row(line: str) -> list[str]:
    """Parse one row from commu_meta.csv.

    The file stores each row as a quoted Python-style nested list, for example:
    "[['Am', 'Am', 'C', 'C']]"
    """
    obj = ast.literal_eval(line.strip())
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], list):
        obj = obj[0]
    if not isinstance(obj, list) or not all(isinstance(chord, str) for chord in obj):
        raise ValueError(f"Unexpected chord row format: {line[:80]!r}")
    return obj


def load_chord_sequences(meta_path: Path) -> list[list[str]]:
    sequences = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                sequences.append(parse_chord_row(line))
            except Exception as exc:
                raise ValueError(f"Could not parse {meta_path}:{line_number}") from exc
    return sequences


def build_chord_vocab(sequences: list[list[str]]) -> dict[str, int]:
    counts = Counter(chord for seq in sequences for chord in seq)
    vocab = {PAD: 0, UNK: 1}
    for chord, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        vocab[chord] = len(vocab)
    return vocab


def build_melody_vocab() -> dict[str, int]:
    vocab = {PAD: 0, START: 1, REST: 2, HOLD: 3}
    for pitch in range(128):
        vocab[f"NOTE_{pitch}"] = len(vocab)
    return vocab


def find_midi_path(midi_root: Path, index: int) -> tuple[str, Path]:
    filename = f"commu{index:05d}.mid"
    matches = list(midi_root.glob(f"*/raw/{filename}"))
    if not matches:
        matches = list(midi_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {midi_root}")
    path = matches[0]
    split = path.parts[-3] if len(path.parts) >= 3 else "unknown"
    return split, path


def midi_to_melody_tokens(midi_path: Path, steps: int) -> list[str]:
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is required for MIDI preprocessing. Install it with:\n"
            "  pip install pretty_midi"
        ) from exc

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    instruments = [inst for inst in midi.instruments if not inst.is_drum and inst.notes]
    if not instruments:
        return [REST] * steps

    instrument = max(instruments, key=lambda inst: len(inst.notes))
    end_time = midi.get_end_time()
    if end_time <= 0:
        return [REST] * steps

    step_size = end_time / steps
    tokens = []
    notes = sorted(instrument.notes, key=lambda note: note.start)
    previous_pitch = None

    for step in range(steps):
        start = step * step_size
        end = (step + 1) * step_size
        active = [note.pitch for note in notes if note.start < end and note.end > start]
        if not active:
            tokens.append(REST)
            previous_pitch = None
            continue

        pitch = max(active)
        if pitch == previous_pitch:
            tokens.append(HOLD)
        else:
            tokens.append(f"NOTE_{pitch}")
        previous_pitch = pitch

    return tokens


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preprocess(args: argparse.Namespace) -> None:
    meta_path = Path(args.meta)
    midi_root = Path(args.midi_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chord_sequences = load_chord_sequences(meta_path)
    if args.max_items is not None:
        chord_sequences = chord_sequences[: args.max_items]

    chord_vocab = build_chord_vocab(chord_sequences)
    melody_vocab = build_melody_vocab()

    write_json(out_dir / "chord_vocab.json", chord_vocab)
    write_json(out_dir / "melody_vocab.json", melody_vocab)

    writers = {}
    stats = Counter()
    try:
        for row_index, chords in enumerate(chord_sequences, start=1):
            split, midi_path = find_midi_path(midi_root, row_index)
            out_path = out_dir / f"{split}.jsonl"
            if split not in writers:
                writers[split] = out_path.open("w", encoding="utf-8")

            chord_ids = [chord_vocab.get(chord, chord_vocab[UNK]) for chord in chords]

            if args.chords_only:
                melody_ids = []
                melody_input_ids = []
            else:
                melody_tokens = midi_to_melody_tokens(midi_path, steps=len(chords))
                melody_ids = [melody_vocab[token] for token in melody_tokens]
                melody_input_ids = [melody_vocab[START]] + melody_ids[:-1]

            example = {
                "id": f"commu{row_index:05d}",
                "split": split,
                "midi_path": str(midi_path),
                "length": len(chords),
                "chord_ids": chord_ids,
                "melody_input_ids": melody_input_ids,
                "melody_target_ids": melody_ids,
            }
            writers[split].write(json.dumps(example) + "\n")
            stats[split] += 1
    finally:
        for writer in writers.values():
            writer.close()

    summary = {
        "examples": sum(stats.values()),
        "splits": dict(stats),
        "chord_vocab_size": len(chord_vocab),
        "melody_vocab_size": len(melody_vocab),
        "melody_tokens": {
            "pad": PAD,
            "start": START,
            "rest": REST,
            "hold": HOLD,
        }
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chord and melody token ID files for chord-conditioned melody generation."
    )
    parser.add_argument("--meta", default=str(BASE_DIR / "dataset" / "commu_meta.csv"))
    parser.add_argument("--midi-root", default=str(BASE_DIR / "dataset" / "commu_midi"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "preprocessed"))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--chords-only",
        action="store_true",
        help="Skip MIDI parsing and only write chord IDs. Useful before installing pretty_midi.",
    )
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
