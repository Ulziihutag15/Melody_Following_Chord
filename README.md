# Melody Following Chord

Chord-conditioned melody generation using the ComMU chord progressions and MIDI
files. The model uses a Transformer encoder-decoder:

- chord progression tokens go into the encoder
- previous melody tokens go into the decoder
- the decoder predicts the next melody token

The current melody vocabulary includes `REST`, `HOLD`, and `NOTE_0` through
`NOTE_127`.

## Project Layout

```text
midi_generation/
  dataset/
    commu_meta.csv
    commu_midi/          # ignored by git
  model/
    chord_features.py
    encoder.py
    decoder.py
    seq2seq.py
  preprocess.py
  train.py
  generate.py
```

Generated files are ignored by git:

```text
midi_generation/dataset/commu_midi/
midi_generation/preprocessed/
midi_generation/checkpoints/
midi_generation/outputs/
```

## Setup

Install dependencies:

```bash
pip install torch pretty_midi
```

CD into the project directory:

```bash
cd midi_generation
```

extract `commu_midi.tar`:

```bash
tar -xf dataset/commu_midi.tar -C midi_generation/dataset/
```

Expected dataset layout:

```text
midi_generation/dataset/commu_meta.csv
midi_generation/dataset/commu_midi/train/raw/*.mid
midi_generation/dataset/commu_midi/val/raw/*.mid
```

## Preprocess

Build chord and melody token files:

```bash
python preprocess.py
```

This creates:

```text
midi_generation/preprocessed/chord_vocab.json
midi_generation/preprocessed/melody_vocab.json
midi_generation/preprocessed/train.jsonl
midi_generation/preprocessed/val.jsonl
```

## Train

Train the encoder-decoder model:

```bash
python train.py --epochs 50 --batch-size 32
```

Checkpoints are saved to:

```text
midi_generation/checkpoints/best.pt
midi_generation/checkpoints/last.pt
```

`best.pt` is the checkpoint with the lowest validation loss.

## Generate

Generate from a compact chord progression:

```bash
python generate.py Am-G-F-E-Am-G-F-E --steps-per-chord 8
```

This writes:

```text
midi_generation/outputs/generated.mid
```

If you provide a full time-step chord sequence, omit `--steps-per-chord`:

```bash
python generate.py Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G
```

Example progressions:

```bash
python generate.py Am-G-F-E-Am-G-F-E --steps-per-chord 8
python generate.py C-G-Am-F-C-G-Am-F --steps-per-chord 8
python generate.py G-D-Em-C-G-D-Em-C --steps-per-chord 8
python generate.py Am-F-C-G-Am-F-C-G --steps-per-chord 8
```

## Notes

`--steps-per-chord 8` means each chord lasts 8 model time steps. With the
default MIDI timing, this is roughly one bar at 120 BPM.

The `HOLD` token represents a sustained note, so the model can distinguish
between holding a note and replaying the same pitch.
