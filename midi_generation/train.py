#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.chord_features import build_chord_feature_matrix
from model.seq2seq import ChordConditionedMelodyModel


PAD_ID = 0


class MelodyJsonlDataset(Dataset):
    def __init__(self, path: str | Path, max_items: int | None = None):
        self.examples = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.examples.append(json.loads(line))
                if max_items is not None and len(self.examples) >= max_items:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def pad_sequence(values: list[list[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    max_len = max(len(item) for item in values)
    padded = [item + [pad_id] * (max_len - len(item)) for item in values]
    return torch.tensor(padded, dtype=torch.long)


def collate_batch(examples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "ids": [example["id"] for example in examples],
        "chord_ids": pad_sequence([example["chord_ids"] for example in examples]),
        "melody_input_ids": pad_sequence(
            [example["melody_input_ids"] for example in examples]
        ),
        "melody_target_ids": pad_sequence(
            [example["melody_target_ids"] for example in examples]
        ),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        chord_ids = batch["chord_ids"].to(device)
        melody_input_ids = batch["melody_input_ids"].to(device)
        melody_target_ids = batch["melody_target_ids"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(chord_ids, melody_input_ids)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                melody_target_ids.reshape(-1),
            )

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        token_count = melody_target_ids.ne(PAD_ID).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def load_vocab_size(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return len(json.load(f))


def load_vocab(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a chord-conditioned encoder-decoder melody model."
    )
    parser.add_argument("--train-jsonl", default="preprocessed/train.jsonl")
    parser.add_argument("--val-jsonl", default="preprocessed/val.jsonl")
    parser.add_argument("--chord-vocab", default="preprocessed/chord_vocab.json")
    parser.add_argument("--melody-vocab", default="preprocessed/melody_vocab.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument(
        "--use-chord-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add pitch-class chord features to the learned chord ID embeddings.",
    )
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    chord_vocab = load_vocab(args.chord_vocab)
    chord_vocab_size = len(chord_vocab)
    melody_vocab_size = load_vocab_size(args.melody_vocab)
    chord_feature_matrix = (
        build_chord_feature_matrix(chord_vocab) if args.use_chord_features else None
    )

    train_data = MelodyJsonlDataset(args.train_jsonl, max_items=args.max_train_items)
    val_data = MelodyJsonlDataset(args.val_jsonl, max_items=args.max_val_items)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model = ChordConditionedMelodyModel(
        chord_vocab_size=chord_vocab_size,
        melody_vocab_size=melody_vocab_size,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pad_id=PAD_ID,
        max_len=args.max_len,
        use_chord_features=args.use_chord_features,
        chord_feature_matrix=chord_feature_matrix,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    print(f"device: {device}")
    print(f"train examples: {len(train_data)}")
    print(f"val examples: {len(val_data)}")
    print(f"chord vocab: {chord_vocab_size}")
    print(f"melody vocab: {melody_vocab_size}")

    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f}"
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            args,
            epoch,
            train_loss,
            val_loss,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                args,
                epoch,
                train_loss,
                val_loss,
            )


if __name__ == "__main__":
    main()
