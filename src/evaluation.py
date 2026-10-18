import hashlib
import json
import os
import tempfile
import time

import keras
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


class EpochTimer(keras.callbacks.Callback):
    """Records wall-clock seconds per epoch. Pass to model.fit(callbacks=[...])."""

    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._start)

    @property
    def median(self):
        return float(np.median(self.epoch_times)) if self.epoch_times else None


def model_size_mb(model):
    """Legacy checkpoint size in MiB; a compiled model includes optimizer state.

    New SOTA reports use model_reporting_metrics to distinguish training
    checkpoints from optimizer-free inference files.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.keras")
        model.save(path)
        return os.path.getsize(path) / (1024 ** 2)


def estimate_flops(model):
    """Forward-pass FLOPs for one image (batch size 1). ~2 x MACs.

    Returns None if the TF profiler cannot trace the model.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        spec = tf.TensorSpec([1, *model.input_shape[1:]], tf.float32)
        concrete = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            builder = tf.compat.v1.profiler.ProfileOptionBuilder
            opts = builder(builder.float_operation()).with_empty_output().build()
            prof = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts
            )
        return int(prof.total_float_ops)
    except Exception as e:  # profiler support varies across TF/Keras versions
        print(f"[evaluation] FLOPs estimate unavailable: {e}")
        return None


_FLOPS_NOT_PROVIDED = object()


def estimate_macs(model, flops=_FLOPS_NOT_PROVIDED):
    """Estimate multiply-accumulate operations per image from the FLOPs count.

    For convolution- and dense-heavy models, one multiply-accumulate is often
    counted as about two floating-point operations (multiply + add). This is
    an approximation; profiler conventions and other operations can differ.
    """
    # Pass an existing FLOPs result when you already measured it, avoiding a
    # second TensorFlow trace/profiling run.
    if flops is _FLOPS_NOT_PROVIDED:
        flops = estimate_flops(model)
    return int(flops // 2) if flops is not None else None


def measure_cpu_latency_ms(model, warmup_runs=10, measured_runs=100):
    """Return median single-image inference latency in milliseconds on CPU.

    The helper clones the trained model and copies its weights inside a CPU
    device scope, so a model trained on GPU is still measured on CPU. The
    synthetic all-zero input has the model's input shape; no test images are
    needed for a timing measurement.
    """
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be zero or greater")
    if measured_runs < 1:
        raise ValueError("measured_runs must be at least one")

    # This helper expects one fixed-shape image input, as used by the project models.
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        raise ValueError("CPU latency helper expects a model with one input")
    sample_shape = tuple(input_shape[1:])
    if any(dim is None for dim in sample_shape):
        raise ValueError("CPU latency helper expects fixed non-batch input dimensions")

    with tf.device("/CPU:0"):
        # Cloning inside the CPU scope places the benchmark model on CPU even
        # when the original model was trained on a GPU. Copy over trained weights.
        cpu_model = keras.models.clone_model(model)
        cpu_model.set_weights(model.get_weights())
        sample = tf.zeros((1, *sample_shape), dtype=tf.float32)

        # Trace once and warm up first; exclude tracing/start-up time from results.
        @tf.function
        def predict_one(batch):
            return cpu_model(batch, training=False)

        for _ in range(warmup_runs):
            predict_one(sample).numpy()  # .numpy() waits for CPU work to finish.

        run_times_ms = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            predict_one(sample).numpy()
            run_times_ms.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(run_times_ms))


def macro_precision_recall(y_true, y_pred, num_classes):
    """Macro-averaged precision and recall (classes with no predictions count as 0)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    pred_per_class = cm.sum(axis=0)
    true_per_class = cm.sum(axis=1)
    precision = np.divide(tp, pred_per_class, out=np.zeros_like(tp), where=pred_per_class > 0)
    recall = np.divide(tp, true_per_class, out=np.zeros_like(tp), where=true_per_class > 0)
    return float(precision.mean()), float(recall.mean())


def predict_probabilities(model, ds):
    """Collect matching labels and probabilities in ONE prediction pass."""
    labels, probabilities = [], []
    for images, batch_labels in ds:
        probabilities.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    if not labels:
        raise ValueError("Cannot evaluate an empty dataset")
    return np.concatenate(labels).astype(int), np.concatenate(probabilities)


def predict_labels(model, ds):
    """Return matching labels and predicted classes from one dataset pass."""
    labels, probabilities = predict_probabilities(model, ds)
    return labels, np.argmax(probabilities, axis=1)


def model_reporting_metrics(model, training_checkpoint, inference_file):
    """Use the same size, computation and CPU timing measures for both SOTAs.

    inference_file must have been saved from a model loaded with compile=False.
    Keep the original compiled checkpoint for resuming training. MiB is always
    bytes / 1024**2; neither file size is the model's peak runtime RAM usage.
    """
    flops = estimate_flops(model)
    try:
        latency = measure_cpu_latency_ms(model)
    except Exception as error:
        print(f"CPU latency measurement unavailable: {error}")
        latency = None
    total = int(model.count_params())
    trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "training_checkpoint_size_mib": Path(training_checkpoint).stat().st_size / 1024**2,
        "inference_model_size_mib": Path(inference_file).stat().st_size / 1024**2,
        "weight_data_size_mib": sum(weight.numpy().nbytes for weight in model.weights) / 1024**2,
        "flops_per_image": flops,
        "mflops_per_image": flops / 1e6 if flops is not None else None,
        "estimated_macs_per_image": estimate_macs(model, flops=flops),
        "cpu_inference_median_ms_per_image_batch1": latency,
        "cpu_latency_warmup_runs": 10,
        "cpu_latency_measured_runs": 100,
    }


def dataset_digest(path):
    """Record which processed split file a run used, for consistent final testing."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_classification_report(model, dataset, class_names, output_dir, split,
                               return_predictions=False):
    """Save scores/plots from one pass; optionally return labels/predictions too.

    Returning predictions lets stage-comparison plots reuse the same pass.
    The normal return value remains (metrics, class_report, confusion_matrix).
    """
    import csv
    from src import plots
    from src.utils import save_json

    if split not in {"validation", "test"}:
        raise ValueError("Reporting split must be validation or test")
    labels, probabilities = predict_probabilities(model, dataset)
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Model outputs do not match the supplied class names")
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    scores = _classification_metrics(confusion)
    per_class = scores.pop("per_class")
    metrics = {
        "split": split, "images": int(len(labels)),
        f"{split}_loss": float(keras.losses.SparseCategoricalCrossentropy()(labels, probabilities).numpy()),
        f"{split}_accuracy": float(np.mean(labels == predictions)),
        **scores,
    }
    report = {
        name: {"precision": item["precision"], "recall": item["recall"],
               "f1": item["f1_score"], "support": item["support"]}
        for name, item in zip(class_names, per_class)
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out / f"{split}_metrics.json")
    save_json(report, out / f"{split}_classification_report.json")
    np.savetxt(out / f"{split}_confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    with (out / f"{split}_confusion_matrix_labeled.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted", *class_names])
        for name, row in zip(class_names, confusion):
            writer.writerow([name, *row.tolist()])
    plots.plot_confusion_matrix(labels, predictions, class_names, str(out / "figures"), f"{split} confusion matrix")
    plots.plot_per_class_metrics(labels, predictions, class_names, str(out / "figures"), f"{split} scores by class")
    if return_predictions:
        return metrics, report, confusion, labels, predictions
    return metrics, report, confusion


def collect_metrics(model, test_ds, timer, num_classes=10, save_path=None):
    """Compute all reporting metrics for a trained model and optionally save as JSON."""
    y_true, y_pred = predict_labels(model, test_ds)

    precision, recall = macro_precision_recall(y_true, y_pred, num_classes)
    flops = estimate_flops(model)

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())

    results = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "model_size_mb": round(model_size_mb(model), 3),
        "flops_per_image": flops,
        "mflops_per_image": round(flops / 1e6, 2) if flops else None,
        "test_accuracy": float((y_true == y_pred).mean()),
        "macro_precision": precision,
        "macro_recall": recall,
        "median_epoch_time_s": timer.median,
        "epoch_times_s": [round(t, 3) for t in timer.epoch_times],
    }

    for k, v in results.items():
        if k != "epoch_times_s":
            print(f"{k:28s} {v}")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {save_path}")
    return results
def _classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    true_positive = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    support = actual.astype(int)
    return {
        "per_class": [
            {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(true_positive))
        ],
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }


def evaluate_classifier(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    output_dir: str | Path,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a classifier and save metrics, confusion matrix, and figures."""
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in test_dataset])
    probabilities = model.predict(test_dataset, verbose=0)
    predictions = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    confusion = tf.math.confusion_matrix(labels, predictions, num_classes=num_classes).numpy()
    metrics = _classification_metrics(confusion)
    metrics["confusion_matrix"] = confusion.tolist()
    metrics["class_names"] = class_names or [str(index) for index in range(num_classes)]
    np.save(output_path / "confusion_matrix.npy", confusion)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Model A confusion matrix")
    axis.set_xticks(range(num_classes), metrics["class_names"], rotation=45, ha="right")
    axis.set_yticks(range(num_classes), metrics["class_names"])
    for row in range(num_classes):
        for column in range(num_classes):
            axis.text(column, row, int(confusion[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics


def plot_history(history: dict[str, list[float]], output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    if not history.get("loss"):
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path / "training_curves.png", dpi=160)
    plt.close(figure)
