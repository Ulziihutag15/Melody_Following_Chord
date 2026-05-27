#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

try:
    from .model.chord_features import build_chord_feature_matrix
    from .model.seq2seq import ChordConditionedMelodyModel
except ImportError:
    from model.chord_features import build_chord_feature_matrix
    from model.seq2seq import ChordConditionedMelodyModel


PAD_ID = 0
BASE_DIR = Path(__file__).resolve().parent
START = "<START>"
REST = "REST"
HOLD = "HOLD"

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


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {idx: token for token, idx in vocab.items()}


def parse_chord(chord: str) -> tuple[str, str]:
    if len(chord) >= 2 and chord[1] in {"#", "b"}:
        return chord[:2], chord[2:]
    return chord[:1], chord[1:]


def chord_pitch_classes(chord: str) -> set[int]:
    root, quality = parse_chord(chord)
    root_pc = ROOT_TO_PC.get(root)
    intervals = QUALITY_INTERVALS.get(quality)
    if root_pc is None or intervals is None:
        return set()
    return {(root_pc + interval) % 12 for interval in intervals}


def build_note_id_maps(melody_vocab: dict[str, int]) -> tuple[dict[int, int], dict[int, int]]:
    id_to_pitch = {}
    pitch_to_id = {}
    for token, token_id in melody_vocab.items():
        if token.startswith("NOTE_"):
            pitch = int(token.split("_", 1)[1])
            id_to_pitch[token_id] = pitch
            pitch_to_id[pitch] = token_id
    return id_to_pitch, pitch_to_id


def build_chord_tone_ids(
    chord: str,
    pitch_to_id: dict[int, int],
    min_pitch: int,
    max_pitch: int,
) -> set[int]:
    pitch_classes = chord_pitch_classes(chord)
    if not pitch_classes:
        return set()
    return {
        token_id
        for pitch, token_id in pitch_to_id.items()
        if min_pitch <= pitch <= max_pitch and pitch % 12 in pitch_classes
    }


def expand_chords(chords: list[str], steps_per_chord: int) -> list[str]:
    expanded = []
    for chord in chords:
        expanded.extend([chord] * steps_per_chord)
    return expanded


def normalize_chord_args(chords: list[str]) -> list[str]:
    normalized = []
    for item in chords:
        parts = [part.strip() for part in item.replace(",", "-").split("-")]
        normalized.extend(part for part in parts if part)
    return normalized


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_model(
    checkpoint: dict,
    chord_vocab: dict[str, int],
    melody_vocab_size: int,
) -> ChordConditionedMelodyModel:
    args = checkpoint.get("args", {})
    use_chord_features = args.get("use_chord_features", False)
    chord_feature_matrix = (
        build_chord_feature_matrix(chord_vocab) if use_chord_features else None
    )
    return ChordConditionedMelodyModel(
        chord_vocab_size=len(chord_vocab),
        melody_vocab_size=melody_vocab_size,
        d_model=args.get("d_model", 128),
        encoder_layers=args.get("encoder_layers", 4),
        decoder_layers=args.get("decoder_layers", 4),
        num_heads=args.get("num_heads", 4),
        dim_feedforward=args.get("dim_feedforward", 512),
        dropout=args.get("dropout", 0.1),
        pad_id=PAD_ID,
        max_len=args.get("max_len", 1024),
        use_chord_features=use_chord_features,
        chord_feature_matrix=chord_feature_matrix,
    )


def banned_ngram_tokens(melody_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 1 or len(melody_ids) < ngram_size - 1:
        return set()

    prefix = tuple(melody_ids[-(ngram_size - 1):])
    banned = set()
    for idx in range(len(melody_ids) - ngram_size + 1):
        ngram = tuple(melody_ids[idx: idx + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    melody_ids: list[int],
    rest_id: int,
    id_to_pitch: dict[int, int],
    min_pitch: int,
    max_pitch: int,
    chord_tone_ids: set[int],
    chord_tone_bias: float,
    repetition_penalty: float,
    immediate_repeat_penalty: float,
    max_repeat: int,
    no_repeat_ngram_size: int,
) -> int:
    logits = logits.clone()

    for token_id, pitch in id_to_pitch.items():
        if pitch < min_pitch or pitch > max_pitch:
            logits[token_id] = float("-inf")

    if chord_tone_bias != 0 and chord_tone_ids:
        for token_id in chord_tone_ids:
            logits[token_id] = logits[token_id] + chord_tone_bias

    if melody_ids and repetition_penalty > 1.0:
        for token_id in set(melody_ids[-16:]):
            if token_id != rest_id:
                logits[token_id] = logits[token_id] / repetition_penalty

    if melody_ids and immediate_repeat_penalty > 1.0 and melody_ids[-1] != rest_id:
        logits[melody_ids[-1]] = logits[melody_ids[-1]] / immediate_repeat_penalty

    if max_repeat > 0 and len(melody_ids) >= max_repeat:
        recent = melody_ids[-max_repeat:]
        if len(set(recent)) == 1 and recent[0] != rest_id:
            logits[recent[0]] = float("-inf")

    for token_id in banned_ngram_tokens(melody_ids, no_repeat_ngram_size):
        if token_id != rest_id:
            logits[token_id] = float("-inf")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_melody_ids(
    model: ChordConditionedMelodyModel,
    chord_ids: list[int],
    expanded_chords: list[str],
    start_id: int,
    rest_id: int,
    id_to_pitch: dict[int, int],
    pitch_to_id: dict[int, int],
    device: torch.device,
    temperature: float,
    top_k: int,
    min_pitch: int,
    max_pitch: int,
    chord_tone_bias: float,
    repetition_penalty: float,
    immediate_repeat_penalty: float,
    max_repeat: int,
    no_repeat_ngram_size: int,
) -> list[int]:
    model.eval()
    chord_tensor = torch.tensor([chord_ids], dtype=torch.long, device=device)
    melody_ids = [start_id]

    with torch.no_grad():
        for _ in range(len(chord_ids)):
            step = len(melody_ids) - 1
            chord_tone_ids = build_chord_tone_ids(
                expanded_chords[step],
                pitch_to_id,
                min_pitch,
                max_pitch,
            )
            melody_tensor = torch.tensor([melody_ids], dtype=torch.long, device=device)
            logits = model(chord_tensor, melody_tensor)
            next_logits = logits[0, -1]
            next_id = sample_next_token(
                next_logits,
                temperature,
                top_k,
                melody_ids,
                rest_id,
                id_to_pitch,
                min_pitch,
                max_pitch,
                chord_tone_ids,
                chord_tone_bias,
                repetition_penalty,
                immediate_repeat_penalty,
                max_repeat,
                no_repeat_ngram_size,
            )
            melody_ids.append(next_id)

    return melody_ids[1:]


def melody_stats(melody_ids: list[int], rest_id: int, id_to_pitch: dict[int, int]) -> dict:
    note_ids = [token_id for token_id in melody_ids if token_id in id_to_pitch]
    if not note_ids:
        return {
            "num_notes": 0,
            "unique_notes": 0,
            "unique_ratio": 0.0,
            "rest_ratio": 1.0,
            "adjacent_repeat_ratio": 1.0,
            "longest_run": 0,
            "score": -999.0,
        }

    adjacent_repeats = sum(
        1 for prev, curr in zip(note_ids, note_ids[1:]) if prev == curr
    )
    runs = []
    current_run = 1
    for prev, curr in zip(note_ids, note_ids[1:]):
        if prev == curr:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    unique_notes = len(set(note_ids))
    unique_ratio = unique_notes / len(note_ids)
    rest_ratio = melody_ids.count(rest_id) / max(len(melody_ids), 1)
    adjacent_repeat_ratio = adjacent_repeats / max(len(note_ids) - 1, 1)
    pitch_values = [id_to_pitch[token_id] for token_id in note_ids]
    contour_steps = [abs(b - a) for a, b in zip(pitch_values, pitch_values[1:])]
    avg_motion = mean(contour_steps) if contour_steps else 0.0

    score = (
        unique_ratio * 3.0
        - adjacent_repeat_ratio * 0.75
        - rest_ratio * 1.5
        + min(avg_motion, 7.0) * 0.08
    )

    return {
        "num_notes": len(note_ids),
        "unique_notes": unique_notes,
        "unique_ratio": unique_ratio,
        "rest_ratio": rest_ratio,
        "adjacent_repeat_ratio": adjacent_repeat_ratio,
        "longest_run": max(runs),
        "score": score,
    }


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def note_sort_key(note_name: str) -> tuple[int, int]:
    octave = int(note_name[-1])
    pitch_class = note_name[:-1]
    return octave, NOTE_NAMES.index(pitch_class)


def format_pitch_classes(pitch_classes: set[int]) -> str:
    return "/".join(NOTE_NAMES[pc] for pc in sorted(pitch_classes)) or "unknown"


def analyze_chord_fit(
    melody_ids: list[int],
    expanded_chords: list[str],
    id_to_pitch: dict[int, int],
) -> dict:
    note_rows = []
    sections = []

    for step, (token_id, chord) in enumerate(zip(melody_ids, expanded_chords)):
        if not sections or sections[-1]["chord"] != chord:
            sections.append(
                {
                    "chord": chord,
                    "start_step": step,
                    "end_step": step + 1,
                    "notes": [],
                }
            )
        else:
            sections[-1]["end_step"] = step + 1

        if token_id not in id_to_pitch:
            continue

        pitch = id_to_pitch[token_id]
        pitch_classes = chord_pitch_classes(chord)
        in_chord = bool(pitch_classes) and pitch % 12 in pitch_classes
        row = {
            "step": step,
            "chord": chord,
            "pitch": pitch,
            "note": pitch_name(pitch),
            "in_chord": in_chord,
            "tones": format_pitch_classes(pitch_classes),
        }
        note_rows.append(row)
        sections[-1]["notes"].append(row)

    total_notes = len(note_rows)
    chord_tone_rate = (
        sum(row["in_chord"] for row in note_rows) / total_notes if total_notes else 0.0
    )
    pitch_values = [row["pitch"] for row in note_rows]

    return {
        "total_notes": total_notes,
        "unique_notes": sorted({row["note"] for row in note_rows}, key=note_sort_key),
        "pitch_range": (
            (pitch_name(min(pitch_values)), pitch_name(max(pitch_values)))
            if pitch_values
            else None
        ),
        "chord_tone_rate": chord_tone_rate,
        "sections": sections,
        "note_rows": note_rows,
    }


def print_chord_fit_analysis(analysis: dict, show_timeline: bool = False) -> None:
    print("chord-fit analysis:")
    print(f"  notes: {analysis['total_notes']}")
    print(f"  pitch_range: {analysis['pitch_range']}")
    print(f"  unique_notes: {' '.join(analysis['unique_notes'])}")
    print(f"  overall_chord_tone_rate: {analysis['chord_tone_rate']:.2f}")
    print("  sections:")

    for index, section in enumerate(analysis["sections"], start=1):
        notes = section["notes"]
        if notes:
            tone_rate = sum(row["in_chord"] for row in notes) / len(notes)
            note_text = " ".join(
                row["note"] if row["in_chord"] else f"{row['note']}*"
                for row in notes
            )
        else:
            tone_rate = 0.0
            note_text = "(no note onsets)"
        print(
            f"    {index:02d} {section['chord']} "
            f"steps {section['start_step']}-{section['end_step'] - 1}: "
            f"tone_rate={tone_rate:.2f} notes={note_text}"
        )

    if show_timeline:
        print("  timeline (* = non-chord tone):")
        for row in analysis["note_rows"]:
            marker = "" if row["in_chord"] else "*"
            print(
                f"    step {row['step']:02d} {row['chord']:<4s} "
                f"tones({row['tones']}) -> {row['note']}{marker}"
            )


def is_acceptable(stats: dict, min_unique_notes: int, max_adjacent_repeat_ratio: float) -> bool:
    return (
        stats["num_notes"] > 0
        and stats["unique_notes"] >= min_unique_notes
        and stats["adjacent_repeat_ratio"] <= max_adjacent_repeat_ratio
    )


def melody_tokens_to_midi(
    tokens: list[str],
    out_path: str | Path,
    step_seconds: float,
    velocity: int,
) -> None:
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is required to write MIDI. Install it with:\n"
            "  pip install pretty_midi"
        ) from exc

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Generated Melody")

    current_pitch = None
    note_start = 0.0

    def close_note(end_time: float) -> None:
        nonlocal current_pitch, note_start
        if current_pitch is not None and end_time > note_start:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=current_pitch,
                    start=note_start,
                    end=end_time,
                )
            )
        current_pitch = None

    for step, token in enumerate(tokens):
        start = step * step_seconds
        end = (step + 1) * step_seconds

        if token == HOLD:
            if step == len(tokens) - 1:
                close_note(end)
            continue

        if not token.startswith("NOTE_"):
            close_note(start)
            continue

        pitch = int(token.split("_", 1)[1])
        if pitch != current_pitch:
            close_note(start)
            current_pitch = pitch
            note_start = start

        if step == len(tokens) - 1:
            close_note(end)

    midi.instruments.append(instrument)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate melody from a chord progression using a trained checkpoint."
    )
    parser.add_argument(
        "progression",
        nargs="*",
        help="Chord progression, either space-separated or hyphen-separated. Example: Am-G-F-E-Am-G-F-E",
    )
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "checkpoints" / "best.pt"))
    parser.add_argument("--chord-vocab", default=str(BASE_DIR / "preprocessed" / "chord_vocab.json"))
    parser.add_argument("--melody-vocab", default=str(BASE_DIR / "preprocessed" / "melody_vocab.json"))
    parser.add_argument(
        "--chords",
        nargs="+",
        default=None,
        help="Optional named form of the chord progression. Positional input is simpler.",
    )
    parser.add_argument(
        "--steps-per-chord",
        type=int,
        default=None,
        help="Repeat each input chord this many times. If omitted, the input is treated as the full time-step sequence.",
    )
    parser.add_argument(
        "--already-expanded",
        action="store_true",
        help="Treat the chord progression as the full time-step sequence. This is also the default when --steps-per-chord is omitted.",
    )
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-pitch", type=int, default=48)
    parser.add_argument("--max-pitch", type=int, default=84)
    parser.add_argument(
        "--chord-tone-bias",
        type=float,
        default=0.8,
        help="Positive values boost notes that belong to the active chord.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Values above 1.0 reduce recently used non-rest notes.",
    )
    parser.add_argument(
        "--immediate-repeat-penalty",
        type=float,
        default=1.0,
        help="Values above 1.0 strongly reduce choosing the previous note again.",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=0,
        help="Prevent the same non-rest token from appearing this many times in a row. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Prevent repeated note-token patterns of this length. Use 0 to disable.",
    )
    parser.add_argument("--num-generate", type=int, default=1)
    parser.add_argument(
        "--attempts-per-output",
        type=int,
        default=12,
        help="Generate several candidates and keep the best acceptable one.",
    )
    parser.add_argument("--min-unique-notes", type=int, default=5)
    parser.add_argument("--max-adjacent-repeat-ratio", type=float, default=0.6)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out-midi", default=str(BASE_DIR / "outputs" / "generated.mid"))
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument("--velocity", type=int, default=90)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print chord-tone fit and per-section note analysis for each generated MIDI.",
    )
    parser.add_argument(
        "--analysis-timeline",
        action="store_true",
        help="With --analyze, also print every note onset with its chord and chord tones.",
    )
    args = parser.parse_args()

    chord_args = args.chords if args.chords is not None else args.progression
    if not chord_args:
        parser.error("provide a chord progression, for example: Am-G-F-E-Am-G-F-E")

    device = choose_device(args.device)
    chord_vocab = load_json(args.chord_vocab)
    melody_vocab = load_json(args.melody_vocab)
    id_to_melody = invert_vocab(melody_vocab)
    id_to_pitch, pitch_to_id = build_note_id_maps(melody_vocab)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint, chord_vocab, len(melody_vocab)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    input_chords = normalize_chord_args(chord_args)
    if args.already_expanded or args.steps_per_chord is None:
        expanded_chords = input_chords
    else:
        expanded_chords = expand_chords(input_chords, args.steps_per_chord)
    unk_id = chord_vocab["<UNK>"]
    chord_ids = [chord_vocab.get(chord, unk_id) for chord in expanded_chords]

    generated = []
    for output_idx in range(args.num_generate):
        best_candidate = None
        best_stats = None
        for _ in range(args.attempts_per_output):
            melody_ids = generate_melody_ids(
                model=model,
                chord_ids=chord_ids,
                expanded_chords=expanded_chords,
                start_id=melody_vocab[START],
                rest_id=melody_vocab[REST],
                id_to_pitch=id_to_pitch,
                pitch_to_id=pitch_to_id,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                min_pitch=args.min_pitch,
                max_pitch=args.max_pitch,
                chord_tone_bias=args.chord_tone_bias,
                repetition_penalty=args.repetition_penalty,
                immediate_repeat_penalty=args.immediate_repeat_penalty,
                max_repeat=args.max_repeat,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            stats = melody_stats(melody_ids, melody_vocab[REST], id_to_pitch)
            if best_stats is None or stats["score"] > best_stats["score"]:
                best_candidate = melody_ids
                best_stats = stats
            if is_acceptable(
                stats,
                args.min_unique_notes,
                args.max_adjacent_repeat_ratio,
            ):
                break

        assert best_candidate is not None and best_stats is not None
        melody_tokens = [id_to_melody[token_id] for token_id in best_candidate]

        out_midi = Path(args.out_midi)
        if args.num_generate > 1:
            out_midi = out_midi.with_name(
                f"{out_midi.stem}_{output_idx:03d}{out_midi.suffix}"
            )

        melody_tokens_to_midi(
            melody_tokens,
            out_midi,
            step_seconds=args.step_seconds,
            velocity=args.velocity,
        )
        generated.append((out_midi, melody_tokens, best_stats))

    print(f"device: {device}")
    print(f"input chords: {' '.join(input_chords)}")
    print(f"expanded steps: {len(expanded_chords)}")
    for out_midi, melody_tokens, stats in generated:
        print(f"generated tokens: {' '.join(melody_tokens)}")
        print(
            "stats: "
            f"unique_notes={stats['unique_notes']} "
            f"adjacent_repeat_ratio={stats['adjacent_repeat_ratio']:.2f} "
            f"rest_ratio={stats['rest_ratio']:.2f} "
            f"score={stats['score']:.2f}"
        )
        if args.analyze:
            analysis = analyze_chord_fit(
                [melody_vocab[token] for token in melody_tokens],
                expanded_chords,
                id_to_pitch,
            )
            print_chord_fit_analysis(analysis, args.analysis_timeline)


if __name__ == "__main__":
    main()
