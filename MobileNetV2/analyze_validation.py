"""Inspect a saved MobileNetV2 checkpoint using validation data, without training."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyze_validation(checkpoint_path, output_dir=None, batch_size=8):
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    # Import here so --help also works on a computer without TensorFlow.
    import keras
    import numpy as np
    from src import plots
    from src.data_loader import make_tf_datasets

    # 1. Load existing weights. compile=False avoids loading the training optimizer.
    # Read the input size from this checkpoint, not from the current model.py.
    model = keras.models.load_model(str(checkpoint), compile=False)
    shape = model.input_shape
    if (not isinstance(shape, tuple) or len(shape) != 4
            or shape[1] is None or shape[1] != shape[2] or shape[3] != 3):
        raise ValueError("Expected a model with one fixed, square RGB input")
    input_size = int(shape[1])
    class_names = plots.load_class_names()
    if model.output_shape[-1] != len(class_names):
        raise ValueError("Checkpoint output classes do not match the dataset metadata")

    # 2. Reuse the SAME resizing/preprocessing pipeline as training.
    # The shared loader returns three datasets; only val_ds is evaluated here.
    # Pixels remain 0..255 because the saved model already contains Rescaling.
    _, val_ds, _ = make_tf_datasets(
        batch_size=batch_size, sota=True, size=input_size,
    )

    # 3. Collect predictions and their matching labels in one validation pass.
    # training=False disables dropout. There is no fit() or weight update.
    true_batches, probability_batches = [], []
    for images, labels in val_ds:
        probability_batches.append(model(images, training=False).numpy())
        true_batches.append(labels.numpy())
    y_true = np.concatenate(true_batches).astype(int)
    y_prob = np.concatenate(probability_batches)
    y_pred = np.argmax(y_prob, axis=1)

    # 4. Calculate overall and per-class validation scores.
    precision, recall, f1 = plots.per_class_scores(
        y_true, y_pred, len(class_names),
    )
    metrics = {
        "split": "validation",
        "checkpoint": str(checkpoint),
        "input_size": input_size,
        "validation_images": int(len(y_true)),
        "validation_loss": float(
            keras.losses.SparseCategoricalCrossentropy()(y_true, y_prob).numpy()
        ),
        "validation_accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }
    report = {
        name: {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(np.sum(y_true == i)),
        }
        for i, name in enumerate(class_names)
    }

    # 5. Save NEW validation reports beside the checkpoint's existing results.
    out = (Path(output_dir).resolve() if output_dir is not None
           else checkpoint.parent / "validation_analysis")
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("validation_metrics.json", metrics),
        ("validation_classification_report.json", report),
    ):
        (out / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")

    cm = plots.plot_confusion_matrix(
        y_true, y_pred, class_names, str(out / "figures"),
        "MobileNetV2 validation confusion matrix",
    )
    plots.plot_per_class_metrics(
        y_true, y_pred, class_names, str(out / "figures"),
        "MobileNetV2 validation scores by class",
    )
    # Raw CSV has exactly 10 rows and 10 columns, with no header.
    np.savetxt(out / "validation_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    # A second CSV includes class names to make true/predicted labels clear.
    with (out / "validation_confusion_matrix_labeled.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted", *class_names])
        for name, row in zip(class_names, cm):
            writer.writerow([name, *row.tolist()])

    # 6. Reuse the saved training history if it is beside this checkpoint.
    # This plots recorded epochs; it does not train the model again.
    history_path = checkpoint.parent / "stage2_history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        (out / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        plots.plot_learning_curves(history, str(out / "figures"), "Stage 2 history")
    else:
        print(f"History not found: {history_path}; validation reports still saved.")

    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    confusions = sorted(
        [(int(cm[i, j]), i, j) for i in range(len(class_names))
         for j in range(len(class_names)) if i != j and cm[i, j] > 0],
        reverse=True,
    )
    print("\nMost frequent validation mistakes (true class -> predicted class):")
    for count, true_class, predicted_class in confusions[:10]:
        print(f"{class_names[true_class]} -> {class_names[predicted_class]}: {count}")
    print(f"\nReports saved to: {out}")
    return metrics, report, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Saved .keras model path")
    parser.add_argument("--output-dir", help="Default: validation_analysis beside checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    analyze_validation(args.checkpoint, args.output_dir, args.batch_size)
