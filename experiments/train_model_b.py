"""Train Model B on the shared Fashion-MNIST 70/15/15 split.

Run from the repo root:
    python -m experiments.train_model_b --epochs 60

Outputs:
    results/models/model_b.keras            best checkpoint by val_loss (git-ignored)
    results/metrics/model_b_history.json    per-epoch loss/accuracy and epoch times
"""
import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np

from src.data_loader import make_tf_datasets
from src.model_b import build_model_b

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "models" / "model_b.keras"
HISTORY_PATH = ROOT / "results" / "metrics" / "model_b_history.json"


class EpochTimer(keras.callbacks.Callback):
    """Records wall-clock seconds per epoch."""

    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seeds python, numpy and TF so runs are repeatable.
    keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, _ = make_tf_datasets(batch_size=args.batch_size, shuffle_seed=args.seed)

    model = build_model_b()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",   # labels are integer class ids
        metrics=["accuracy"],
    )
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    timer = EpochTimer()
    # Keep the epoch with the lowest validation loss; the test set is never touched here.
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[timer, checkpoint],
        verbose=2,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    record = {
        "model": "model_b",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": f"adam lr={args.lr}",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "epoch_times_s": [round(t, 3) for t in timer.epoch_times],
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Best epoch {best_epoch}; saved model to {MODEL_PATH} and history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
