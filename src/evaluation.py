from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


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