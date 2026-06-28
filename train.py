"""
train.py — Sentinel Model Trainer
──────────────────────────────────
Trains the Sentinel transformer. Training data is read from SQLite
(data/training.db), which is auto-created and seeded on first run.

Usage
-----
    python train.py                        # standard
    python train.py --smoke-test           # quick pipeline check (3 epochs)
    python train.py --reseed               # wipe and re-seed the default examples
    python train.py --db data/training.db  # use a different training database
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("sentinel.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Sentinel intent model")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--vocab-size",  type=int,   default=8192)
    p.add_argument("--checkpoint",  default="checkpoints")
    p.add_argument("--db",          default="data/training.db",
                   help="Path to the training SQLite database")
    p.add_argument("--reseed",      action="store_true",
                   help="Wipe and re-insert default examples before training")
    p.add_argument("--smoke-test",  action="store_true")
    p.add_argument("--debug",       action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.smoke_test:
        log.warning("SMOKE-TEST mode — 3 epochs, batch=8")
        args.epochs     = 3
        args.batch_size = 8

    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    # ── Load from SQLite ──────────────────────────────────────────────
    log.info("Opening training database: %s", args.db)
    from data.training_data import TrainingDB

    db = TrainingDB(args.db)

    if args.reseed:
        log.info("Re-seeding default examples...")
        inserted = db.seed_defaults(force=True)
        log.info("Re-seeded %d examples.", inserted)
    else:
        inserted = db.seed_defaults()
        if inserted:
            log.info("First run — seeded %d default examples.", inserted)

    raw_data      = db.get_all()
    intent_labels = db.get_intent_labels()
    db.close()

    log.info("Dataset: %d examples, %d intents", len(raw_data), len(intent_labels))

    if args.debug or True:
        stats = {}
        for d in raw_data:
            stats[d["intent"]] = stats.get(d["intent"], 0) + 1
        for label in intent_labels:
            log.info("  %-25s  %d examples", label, stats.get(label, 0))

    # ── Train ─────────────────────────────────────────────────────────
    from core.llm.trainer import SentinelTrainer

    trainer = SentinelTrainer(
        output_dir    = args.checkpoint,
        learning_rate = args.lr,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
    )

    log.info("Starting training...  (Ctrl-C to abort)")
    try:
        trainer.train(
            raw_data      = raw_data,
            intent_labels = intent_labels,
            vocab_size    = args.vocab_size,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted — partial checkpoint may exist.")
        return 0
    except Exception:
        log.exception("Training failed")
        return 1

    log.info("Training complete. Checkpoint -> %s/", args.checkpoint)
    log.info("Run: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())