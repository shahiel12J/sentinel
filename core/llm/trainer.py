"""
Sentinel LLM Trainer

Trains the intent-classification head on synthetic command data.
Uses mixed-precision (fp16) for the RTX 3050 (4 GB VRAM).

"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from .model import SentinelTransformer, ModelConfig, build_sentinel_model
from .tokenizer import SentinelTokenizer


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class IntentDataset(Dataset):
    """
    Wraps a list of {text, intent_id} dicts for PyTorch DataLoader.
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: SentinelTokenizer,
        max_length: int = 128,
    ):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        ids, mask = self.tokenizer.encode(
            sample["text"],
            max_length=self.max_length,
            padding=True,
            return_mask=True,
        )
        return {
            "input_ids":       torch.tensor(ids,  dtype=torch.long),
            "padding_mask":    torch.tensor(mask, dtype=torch.bool),
            "label":           torch.tensor(sample["intent_id"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Warm-up cosine schedule
# ─────────────────────────────────────────────

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            import math
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class SentinelTrainer:
    """
    Trains / fine-tunes SentinelTransformer for intent classification.

    Quick-start:
        trainer = SentinelTrainer(output_dir="checkpoints")
        trainer.train(training_data, intent_labels)
    """

    def __init__(
        self,
        output_dir: str = "checkpoints",
        device: Optional[str] = None,
        learning_rate: float = 3e-4,
        weight_decay: float  = 0.01,
        epochs: int          = 20,
        batch_size: int      = 64,
        warmup_ratio: float  = 0.1,
        patience: int        = 5,
        use_amp: bool        = True,
    ):
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.warmup_ratio  = warmup_ratio
        self.patience      = patience
        self.use_amp       = use_amp and self.device.type == "cuda"

        print(f"[Trainer] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Public API ───────────────────────────────────────────────────

    def train(
        self,
        raw_data: List[Dict],          # list of {text, intent} dicts
        intent_labels: List[str],      # ordered list of intent strings
        val_split: float = 0.1,
        vocab_size: int  = 8192,
    ) -> SentinelTransformer:
        """
        Full training pipeline:
          1. Train tokenizer on corpus
          2. Build model
          3. Train with gradient checkpointing + AMP
          4. Save best checkpoint

        Returns the best trained model (also saved to output_dir).
        """

        # ── Step 1: Tokenizer ────────────────────────────────────────
        print("[Trainer] Training tokenizer …")
        tokenizer = SentinelTokenizer()
        tokenizer.train([s["text"] for s in raw_data], vocab_size=vocab_size, verbose=False)
        tok_path = self.output_dir / "tokenizer.json"
        tokenizer.save(str(tok_path))

        # ── Step 2: Encode labels ─────────────────────────────────────
        label_map = {label: i for i, label in enumerate(intent_labels)}
        samples = []
        for s in raw_data:
            if s["intent"] not in label_map:
                continue
            samples.append({"text": s["text"], "intent_id": label_map[s["intent"]]})

        random.shuffle(samples)
        split = max(1, int(len(samples) * val_split))
        val_data, train_data = samples[:split], samples[split:]
        print(f"[Trainer] Train: {len(train_data)}  Val: {len(val_data)}")

        # ── Step 3: Build model ───────────────────────────────────────
        model = build_sentinel_model(
            num_intents=len(intent_labels),
            vocab_size=len(tokenizer),
        )
        print(model.parameter_summary())
        print(f"[Trainer] Est. VRAM (fp16): {model.estimated_vram_mb():.1f} MB")
        model = model.to(self.device)

        # ── Step 4: DataLoaders ───────────────────────────────────────
        train_ds = IntentDataset(train_data, tokenizer)
        val_ds   = IntentDataset(val_data,   tokenizer)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=0)

        # ── Step 5: Optimiser + scheduler ────────────────────────────
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        total_steps  = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler    = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        scaler       = GradScaler(enabled=self.use_amp)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # ── Step 6: Training loop ─────────────────────────────────────
        best_val_acc  = 0.0
        no_improve    = 0
        history       = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch in train_dl:
                ids    = batch["input_ids"].to(self.device)
                pmask  = batch["padding_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    logits = model.classify(ids, pmask)
                    loss   = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss    += loss.item() * labels.size(0)
                train_correct += (logits.argmax(-1) == labels).sum().item()
                train_total   += labels.size(0)

            # Validate
            val_acc, val_loss = self._evaluate(model, val_dl, criterion)

            avg_train_loss = train_loss / train_total
            train_acc      = train_correct / train_total
            elapsed        = time.time() - t0

            print(
                f"Epoch {epoch:>3}/{self.epochs}  "
                f"loss={avg_train_loss:.4f}  acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"[{elapsed:.1f}s]"
            )

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            })

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve   = 0
                self._save_checkpoint(model, tokenizer, intent_labels, history, "best")
                print(f"  ✓ New best val_acc={val_acc:.3f} — checkpoint saved")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"[Trainer] Early stopping (patience={self.patience})")
                    break

        print(f"\n[Trainer] Training complete. Best val_acc={best_val_acc:.3f}")
        return model

    # ── Helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self,
        model: SentinelTransformer,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in dataloader:
            ids    = batch["input_ids"].to(self.device)
            pmask  = batch["padding_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            with autocast(enabled=self.use_amp):
                logits = model.classify(ids, pmask)
                loss   = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(-1) == labels).sum().item()
            total      += labels.size(0)
        model.train()
        return correct / max(1, total), total_loss / max(1, total)

    def _save_checkpoint(
        self,
        model:         SentinelTransformer,
        tokenizer:     SentinelTokenizer,
        intent_labels: List[str],
        history:       List[Dict],
        tag:           str = "best",
    ) -> None:
        path = self.output_dir / f"sentinel_{tag}.pt"
        torch.save(
            {
                "model_state":   model.state_dict(),
                "model_config":  model.config.__dict__,
                "intent_labels": intent_labels,
                "history":       history,
            },
            path,
        )


# ─────────────────────────────────────────────
# Loader (used at runtime)
# ─────────────────────────────────────────────

def load_checkpoint(
    checkpoint_dir: str = "checkpoints",
    device: Optional[str] = None,
) -> Tuple[Optional[SentinelTransformer], Optional[SentinelTokenizer], Optional[List[str]]]:
    """
    Load the best saved checkpoint.
    Returns (model, tokenizer, intent_labels) or (None, None, None) if not found.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(checkpoint_dir) / "sentinel_best.pt"
    tok_path  = Path(checkpoint_dir) / "tokenizer.json"

    if not ckpt_path.exists() or not tok_path.exists():
        return None, None, None

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ModelConfig(**ckpt["model_config"])
    model = SentinelTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tokenizer = SentinelTokenizer.load(str(tok_path))

    return model, tokenizer, ckpt["intent_labels"]
