import argparse
import csv
import json
import sys
import time
from pathlib import Path

import tensorflow as tf

# Make the repository package importable when this file is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import make_tf_datasets
from src.evaluation import evaluate_classifier, plot_history
from src.models import build_model_a
from src.utils import architecture_metrics, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Model A.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "rmsprop", "sgd", "sgd_momentum"],
        default="adam",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the optimizer default learning rate.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for sgd_momentum.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "model_a")
    return parser.parse_args()


def build_optimizer(name, learning_rate, momentum):
    defaults = {"adam": 0.001, "rmsprop": 0.001, "sgd": 0.01, "sgd_momentum": 0.01}
    rate = learning_rate if learning_rate is not None else defaults[name]
    if name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
    elif name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=rate)
    elif name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=rate, momentum=momentum)
    return optimizer, rate


def main():
    total_start = time.perf_counter()
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    output_dir = args.output_dir
    metrics_dir = output_dir / "metrics"
    models_dir = output_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_a()
    save_json(architecture_metrics(model), metrics_dir / "architecture.json")
    model.summary()

    train_ds, val_ds, test_ds = make_tf_datasets(batch_size=args.batch_size)
    optimizer, learning_rate = build_optimizer(args.optimizer, args.learning_rate, args.momentum)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    training_start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, shuffle=False)
    training_seconds = time.perf_counter() - training_start

    model.save(models_dir / "model_a.keras")
    evaluation_start = time.perf_counter()
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    evaluation = evaluate_classifier(
        model,
        test_ds,
        metrics_dir,
        class_names=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    plot_history(history.history, output_dir / "figures")

    with (metrics_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        keys = list(history.history)
        writer.writerow(["epoch", *keys])
        for index, values in enumerate(zip(*(history.history[key] for key in keys)), start=1):
            writer.writerow([index, *values])

    evaluation.update(
        {
            "model": "model_a",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": learning_rate,
            "momentum": args.momentum if args.optimizer == "sgd_momentum" else 0.0,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "training_seconds": round(training_seconds, 3),
            "evaluation_seconds": round(evaluation_seconds, 3),
            "total_run_seconds": round(time.perf_counter() - total_start, 3),
        }
    )
    (metrics_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Saved model and metrics under: {output_dir}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()