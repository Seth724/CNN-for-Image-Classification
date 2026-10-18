"""Train and evaluate Model B with the same protocol and outputs as scripts/train_model_a.py.

Each run writes to its own --output-dir:
    models/model_b.keras
    metrics/evaluation.json, history.csv, architecture.json, confusion_matrix.png
    figures/training_curves.png

On top of Model A's fields, evaluation.json records total_parameters, total_macs,
final/best validation accuracy (pick configurations on these, not on test accuracy)
and CPU latency. metrics/reporting_metrics.json holds the shared Q6 measurements
(model_reporting_metrics: inference size, profiler MACs, CPU latency).

Q6 (inputs matched to the pretrained models): --input-size 64 upscales with the same
tf.image.resize as the SOTA loader (1 channel, 0-1), and --best-checkpoint evaluates the
epoch with the lowest validation loss, as the SOTA scripts do. The defaults (28x28,
final epoch) reproduce the Q4 runs.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Make the repository package importable when this file is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import make_tf_datasets, make_upscaled_grayscale_datasets
from src.evaluation import (EpochTimer, evaluate_classifier, measure_cpu_latency_ms,
                            model_reporting_metrics, plot_history)
from src.models.model_b import build_model_b
from src.utils import architecture_metrics, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Model B.")
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
    parser.add_argument("--input-size", type=int, default=28,
                        help="28 = native Fashion-MNIST; 64 = upscaled like the pretrained models' input.")
    parser.add_argument("--best-checkpoint", action="store_true",
                        help="Evaluate the epoch with the lowest validation loss instead of the final epoch.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "model_b")
    return parser.parse_args()


def build_optimizer(name, learning_rate, momentum):
    # Same defaults as Model A so optimizer comparisons line up.
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

    size = args.input_size
    model = build_model_b(input_shape=(size, size, 1))
    architecture = architecture_metrics(model)
    save_json(architecture, metrics_dir / "architecture.json")
    model.summary()

    if size == 28:
        train_ds, val_ds, test_ds = make_tf_datasets(batch_size=args.batch_size)
    else:
        train_ds, val_ds, test_ds = make_upscaled_grayscale_datasets(
            batch_size=args.batch_size, size=size, shuffle_seed=args.seed)
    optimizer, learning_rate = build_optimizer(args.optimizer, args.learning_rate, args.momentum)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    training_start = time.perf_counter()
    epoch_timer = EpochTimer()
    callbacks = [epoch_timer]
    best_path = models_dir / "model_b_best.keras"
    if args.best_checkpoint:
        # Same selection rule as the SOTA scripts: lowest validation loss.
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True))
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        shuffle=False,
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - training_start

    # By default, like Model A, the model after the final epoch is saved and evaluated.
    model_path = models_dir / "model_b.keras"
    model.save(model_path)
    if args.best_checkpoint:
        training_checkpoint = best_path
        model = tf.keras.models.load_model(best_path)
        selected_epoch = int(np.argmin(history.history["val_loss"])) + 1
    else:
        training_checkpoint = model_path
        selected_epoch = args.epochs
    model_size_kb = os.path.getsize(training_checkpoint) / 1024
    steady_epoch_times = epoch_timer.epoch_times[1:] or epoch_timer.epoch_times
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
        title="Model B confusion matrix",
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    # Shared Q6 measurements, taken exactly as the SOTA scripts take them: an inference
    # copy saved without optimizer state, profiler MACs and CPU latency.
    inference_model = tf.keras.models.load_model(training_checkpoint, compile=False)
    inference_path = models_dir / "model_b_inference.keras"
    inference_model.save(inference_path)
    reporting = model_reporting_metrics(inference_model, training_checkpoint, inference_path)
    save_json(reporting, metrics_dir / "reporting_metrics.json")
    plot_history(history.history, output_dir / "figures")

    with (metrics_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        keys = list(history.history)
        writer.writerow(["epoch", *keys])
        for index, values in enumerate(zip(*(history.history[key] for key in keys)), start=1):
            writer.writerow([index, *values])

    val_accuracy = history.history["val_accuracy"]
    evaluation.update(
        {
            "model": "model_b",
            "input_size": size,
            "checkpoint": "best_val_loss" if args.best_checkpoint else "final_epoch",
            "selected_epoch": selected_epoch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": learning_rate,
            "momentum": args.momentum if args.optimizer == "sgd_momentum" else 0.0,
            "total_parameters": architecture["total_parameters"],
            "total_macs": architecture["total_macs"],
            # Validation scores are what configurations should be compared on.
            "final_val_accuracy": float(val_accuracy[-1]),
            "best_val_accuracy": float(max(val_accuracy)),
            "best_val_epoch": int(np.argmax(val_accuracy)) + 1,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "model_size_kb": round(model_size_kb, 2),
            "model_size_mb": round(model_size_kb / 1024, 3),
            "mean_epoch_time_s": round(float(sum(steady_epoch_times) / len(steady_epoch_times)), 3),
            "median_epoch_time_s": round(float(statistics.median(epoch_timer.epoch_times)), 3),
            "epoch_times_s": [round(value, 3) for value in epoch_timer.epoch_times],
            "cpu_latency_ms": round(measure_cpu_latency_ms(model), 3),
            "gpu": [device.name for device in tf.config.list_physical_devices("GPU")],
            "training_seconds": round(training_seconds, 3),
            "evaluation_seconds": round(evaluation_seconds, 3),
            "total_run_seconds": round(time.perf_counter() - total_start, 3),
        }
    )
    evaluation["reporting"] = reporting
    (metrics_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Saved model and metrics under: {output_dir}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Final validation accuracy: {val_accuracy[-1]:.4f}")
    print(f"Evaluated epoch: {selected_epoch} ({'best validation loss' if args.best_checkpoint else 'final'})")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
