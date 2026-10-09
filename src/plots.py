"""Report figures shared by all models. Every function saves a PNG and closes the figure.

Typical layout (one folder per training stage, plus a comparison folder):
    results/figures/transfer/    learning curves, confusion matrix, per-class metrics
    results/figures/finetuned/   same set for the fine-tuned model
    results/figures/comparison/  transfer vs fine-tuned, side by side
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # file output only; works headless on Kaggle/Colab
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import predict_labels

COLORS = {"train": "#1f77b4", "val": "#d95f02", "transfer": "#1f77b4", "finetuned": "#2ca02c"}
DPI = 200


def load_class_names(metadata_path=None):
    if metadata_path is None:
        metadata_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "metadata.json"
        )
    with open(metadata_path) as f:
        return json.load(f)["class_names"]


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _history_dict(history):
    return history.history if hasattr(history, "history") else history


def save_history(history, path):
    """Persist a Keras History to JSON so figures can be regenerated without retraining."""
    h = {k: [float(x) for x in v] for k, v in _history_dict(history).items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(h, f, indent=2)


def plot_learning_curves(history, out_dir, title):
    """Loss and accuracy vs epoch (train vs validation) with the best epoch marked."""
    h = _history_dict(history)
    epochs = np.arange(1, len(h["loss"]) + 1)
    acc_key = "sparse_categorical_accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (train_k, val_k, name) in zip(
        axes,
        [("loss", "val_loss", "Loss"), (acc_key, f"val_{acc_key}", "Accuracy")],
    ):
        ax.plot(epochs, h[train_k], "-o", ms=3, color=COLORS["train"], label="Train")
        ax.plot(epochs, h[val_k], "-o", ms=3, color=COLORS["val"], label="Validation")
        best = int(np.argmin(h[val_k]) if name == "Loss" else np.argmax(h[val_k]))
        ax.axvline(best + 1, ls="--", lw=1, color="gray")
        ax.annotate(f"best val: {h[val_k][best]:.4f} (epoch {best + 1})",
                    (best + 1, h[val_k][best]), textcoords="offset points",
                    xytext=(6, 8), fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{name} per epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    _save(fig, os.path.join(out_dir, "learning_curves.png"))


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir, title):
    """Row-normalised confusion matrix (each row sums to 1) with raw counts in the cells."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    _save(fig, os.path.join(out_dir, "confusion_matrix.png"))
    return cm


def per_class_scores(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(n), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n), where=denom > 0)
    return precision, recall, f1


def plot_per_class_metrics(y_true, y_pred, class_names, out_dir, title):
    """Grouped bars of precision / recall / F1 for each class."""
    p, r, f1 = per_class_scores(y_true, y_pred, len(class_names))
    x = np.arange(len(class_names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w, p, w, label="Precision", color="#1f77b4")
    ax.bar(x, r, w, label="Recall", color="#d95f02")
    ax.bar(x + w, f1, w, label="F1", color="#2ca02c")
    ax.set_xticks(x, class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, loc="lower center")
    _save(fig, os.path.join(out_dir, "per_class_metrics.png"))


def plot_epoch_times(epoch_times, out_dir, title):
    """Seconds per epoch with the median drawn as a line."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(np.arange(1, len(epoch_times) + 1), epoch_times, color=COLORS["train"])
    med = float(np.median(epoch_times))
    ax.axhline(med, color="red", ls="--", lw=1, label=f"Median {med:.1f}s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, os.path.join(out_dir, "epoch_times.png"))


def plot_stage_report(model, test_ds, history, epoch_times, class_names, out_dir, stage_name):
    """All per-stage figures for one training stage ('Transfer learning' / 'Fine-tuning')."""
    y_true, y_pred = predict_labels(model, test_ds)
    plot_learning_curves(history, out_dir, f"{stage_name}: learning curves")
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir, f"{stage_name}: confusion matrix (test)")
    plot_per_class_metrics(y_true, y_pred, class_names, out_dir, f"{stage_name}: per-class metrics (test)")
    plot_epoch_times(epoch_times, out_dir, f"{stage_name}: time per epoch")


def plot_full_training_curve(history_transfer, history_finetune, out_dir):
    """Both stages on one timeline, with a line where fine-tuning starts."""
    a, b = _history_dict(history_transfer), _history_dict(history_finetune)
    acc = "sparse_categorical_accuracy"
    n1, n2 = len(a["loss"]), len(b["loss"])
    x = np.arange(1, n1 + n2 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (k, name) in zip(axes, [("loss", "Loss"), (acc, "Accuracy")]):
        ax.plot(x, a[k] + b[k], "-o", ms=3, color=COLORS["train"], label="Train")
        ax.plot(x, a[f"val_{k}"] + b[f"val_{k}"], "-o", ms=3, color=COLORS["val"], label="Validation")
        ax.axvline(n1 + 0.5, ls="--", color="gray")
        ax.text(n1 + 0.7, ax.get_ylim()[0], "fine-tuning starts", fontsize=8, va="bottom")
        ax.set_xlabel("Epoch (continuous across stages)")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Transfer learning -> fine-tuning")
    _save(fig, os.path.join(out_dir, "full_training_curve.png"))


def plot_metric_comparison(transfer_metrics, finetuned_metrics, out_dir):
    """Test accuracy / macro precision / macro recall, transfer vs fine-tuned."""
    keys = [("test_accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
            ("macro_recall", "Macro recall")]
    x = np.arange(len(keys))
    w = 0.36
    t = [transfer_metrics[k] for k, _ in keys]
    f = [finetuned_metrics[k] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, t, w, label="Transfer learning", color=COLORS["transfer"])
    b2 = ax.bar(x + w / 2, f, w, label="Fine-tuned", color=COLORS["finetuned"])
    ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(x, [n for _, n in keys])
    lo = min(t + f)
    ax.set_ylim(max(0, lo - 0.05), 1.0)  # zoomed axis: differences are small, note this in the report
    ax.set_ylabel("Score (y-axis truncated)")
    ax.set_title("Test-set performance")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, os.path.join(out_dir, "metric_comparison.png"))


def plot_per_class_comparison(y_true, pred_transfer, pred_finetuned, class_names, out_dir):
    """Per-class F1 for both stages, to show which classes fine-tuning helped."""
    n = len(class_names)
    f1_t = per_class_scores(y_true, pred_transfer, n)[2]
    f1_f = per_class_scores(y_true, pred_finetuned, n)[2]
    x = np.arange(n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w / 2, f1_t, w, label="Transfer learning", color=COLORS["transfer"])
    ax.bar(x + w / 2, f1_f, w, label="Fine-tuned", color=COLORS["finetuned"])
    ax.set_xticks(x, class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1: transfer learning vs fine-tuned")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", ncol=2)
    _save(fig, os.path.join(out_dir, "per_class_f1_comparison.png"))


def plot_efficiency_table(transfer_metrics, finetuned_metrics, out_dir):
    """Table figure (params, size, FLOPs, epoch time, metrics) ready to paste into a report."""
    rows = [
        ("Total parameters", "total_parameters", "{:,}"),
        ("Trainable parameters", "trainable_parameters", "{:,}"),
        ("Model size (MB)", "model_size_mb", "{:.2f}"),
        ("MFLOPs / image", "mflops_per_image", "{:.2f}"),
        ("Median epoch time (s)", "median_epoch_time_s", "{:.2f}"),
        ("Test accuracy", "test_accuracy", "{:.4f}"),
        ("Macro precision", "macro_precision", "{:.4f}"),
        ("Macro recall", "macro_recall", "{:.4f}"),
    ]
    cells = [[label, fmt.format(transfer_metrics[k]), fmt.format(finetuned_metrics[k])]
             for label, k, fmt in rows]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Metric", "Transfer learning", "Fine-tuned"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    _save(fig, os.path.join(out_dir, "summary_table.png"))
