"""Evaluate the trained Model B on the test set and plot its training curves.

Run after train_model_b.py, from the repo root:
    python -m experiments.evaluate_model_b

Outputs:
    results/metrics/model_b_metrics.json
    results/figures/model_b_loss_curves.png
    results/figures/model_b_confusion_matrix.png
"""
import json
import os
import tempfile
import time
from pathlib import Path

import keras
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.data_loader import make_tf_datasets
from src.model_b import count_macs

matplotlib.use("Agg")   # write figures to files, no window
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "models" / "model_b.keras"
HISTORY_PATH = ROOT / "results" / "metrics" / "model_b_history.json"
METRICS_PATH = ROOT / "results" / "metrics" / "model_b_metrics.json"
FIG_DIR = ROOT / "results" / "figures"
CLASS_NAMES = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text())["class_names"]


def tflite_sizes_kb(model, train_ds):
    """Size of the model exported to TFLite as float32 and as full-int8 (the realistic edge format)."""
    sizes = {}
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        sizes["tflite_float32_kb"] = round(len(conv.convert()) / 1024, 2)

        def representative():
            # A few hundred training images let the converter pick int8 scales.
            for x, _ in train_ds.unbatch().batch(1).take(300):
                yield [x]

        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        sizes["tflite_int8_kb"] = round(len(conv.convert()) / 1024, 2)
    except Exception as e:  # converter support varies across TF versions
        print(f"[evaluate] TFLite export skipped: {e}")
    return sizes


def cpu_latency_ms(model, n=200):
    """Mean time to classify one image (batch size 1), after a warm-up call."""
    x = tf.zeros((1, *model.input_shape[1:]))
    model(x, training=False)
    start = time.perf_counter()
    for _ in range(n):
        model(x, training=False)
    return (time.perf_counter() - start) / n * 1000


def plot_curves(history, best_epoch, path):
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(ax_loss, "loss", "Loss"), (ax_acc, "accuracy", "Accuracy")]:
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_title(f"Model B {title.lower()}")
        ax.grid(alpha=0.3)
        ax.legend()
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_acc.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Model B confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    model = keras.models.load_model(MODEL_PATH)
    run = json.loads(HISTORY_PATH.read_text())
    train_ds, _, test_ds = make_tf_datasets(batch_size=128)

    # Test set is used exactly once, on the best-validation checkpoint.
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Epoch 1 includes graph tracing, so it is excluded from the timing average.
    times = run["epoch_times_s"]
    steady = times[1:] if len(times) > 1 else times
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_b.keras")
        model.save(path)
        size_bytes = os.path.getsize(path)   # includes HDF5/zip container overhead; see tflite sizes for on-device

    metrics = {
        "model": "model_b",
        "total_parameters": int(model.count_params()),
        "trainable_parameters": trainable,
        "non_trainable_parameters": int(model.count_params()) - trainable,
        "macs_per_image": count_macs(model),
        "model_size_kb": round(size_bytes / 1024, 2),
        "model_size_mb": round(size_bytes / 1024 ** 2, 3),
        **tflite_sizes_kb(model, train_ds),
        "test_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "per_class": {
            name: {"precision": round(report[name]["precision"], 4), "recall": round(report[name]["recall"], 4)}
            for name in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist(),
        "mean_epoch_time_s": round(float(np.mean(steady)), 3),
        "median_epoch_time_s": round(float(np.median(times)), 3),
        "epoch_times_s": times,
        "cpu_latency_ms": round(cpu_latency_ms(model), 3),
        "epochs": run["epochs"],
        "best_epoch": run["best_epoch"],
        "optimizer": run["optimizer"],
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(run["history"], run["best_epoch"], FIG_DIR / "model_b_loss_curves.png")
    plot_confusion(cm, FIG_DIR / "model_b_confusion_matrix.png")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        if k not in ("per_class", "confusion_matrix", "epoch_times_s"):
            print(f"{k:26s} {v}")
    print(f"Saved metrics to {METRICS_PATH} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
