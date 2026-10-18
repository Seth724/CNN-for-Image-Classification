import argparse
from datetime import datetime, timezone
import json
import tempfile
import os
from pathlib import Path
import sys
import numpy as np

# Make project imports work when this file is run from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep these choices together so the experiment setup is easy to review.
INPUT_SIZE = 64            # Report the same input resolution for both SOTA models.
NUM_CLASSES = 10           # Fashion-MNIST has ten labels.
BATCH_SIZE = 8
SEED = 42
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 30
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_FRACTION = 0.3   # Target 30% of all backbone layers, skipping BatchNorm.
DROPOUT = 0.2
HEAD_PATIENCE = 3
FINE_TUNE_PATIENCE = 5
ALPHA = 1.0                # Keep the standard MobileNetV2 width for the baseline.

# Each SOTA model must keep its own figures, metrics, logs, and checkpoints.
# Keep the normal local destination, but allow Colab to point outputs to Drive.
# Example: set MOBILENETV2_OUTPUT_DIR before launching this script in Colab.
OUTPUT_BASE = Path(
    os.environ.get(
        "MOBILENETV2_OUTPUT_DIR",
        str(ROOT / "results" / "mobilenetv2_64"),
    )
)


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 using validation data.")
    parser.add_argument("--test_with_test_data", type=Path, metavar="RUN_DIR",
                        help="Evaluate an already selected run on test data; no training")
    args = parser.parse_args()
    if args.test_with_test_data is not None:
        evaluate_saved_run(args.test_with_test_data)
        return

    # Import here so --help and importing this file do not require TensorFlow.
    import keras
    import tensorflow as tf
    from src import plots
    from src.data_loader import DATA_FILE, make_tf_datasets
    from src.evaluation import EpochTimer, dataset_digest, model_reporting_metrics, save_classification_report
    from src.utils import save_json

    # A fixed seed makes data shuffling and Keras randomness more repeatable.
    keras.utils.set_random_seed(SEED)
    # Each execution gets its own folder, preserving earlier results and comments.
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%SZ_")
    OUTPUT_DIR = Path(tempfile.mkdtemp(prefix=stamp, dir=str(OUTPUT_BASE)))
    print(f"Run directory: {OUTPUT_DIR}")
    config = {
        "model_name": "MobileNetV2", "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES, "batch_size": BATCH_SIZE, "seed": SEED,
        "head_epochs": HEAD_EPOCHS, "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
        "head_patience": HEAD_PATIENCE, "fine_tune_patience": FINE_TUNE_PATIENCE,
        "fine_tune_fraction": FINE_TUNE_FRACTION, "dropout": DROPOUT, "alpha": ALPHA,
        "batchnorm_frozen": True, "augmentation": False,
        "selection_metric": "validation_loss", "preprocessing": "Rescaling to [-1, 1] inside model",
        "dataset_sha256": dataset_digest(DATA_FILE),
        "keras_version": keras.__version__, "tensorflow_version": tf.__version__,
    }
    save_json(config, OUTPUT_DIR / "config.json")
    # ----------------------------------------------------------------------
    # TODO 1: Load the shared training and validation splits.
    # ----------------------------------------------------------------------
    # Use the existing make_tf_datasets(batch_size=..., sota=True, size=...).
    # The SOTA loader resizes each batch and converts grayscale images to RGB.
    # Its image values are still in the 0..255 range at this point.
    # Do not create a new split and do not resize the full dataset in memory.
    train_ds, val_ds, _ = make_tf_datasets(
        batch_size=BATCH_SIZE, sota=True, size=INPUT_SIZE, shuffle_seed=SEED,
    )
    # The original loader returns all three splits. Discard the test dataset here.
    # Only train_ds and val_ds are used; no test predictions or scores are computed.
    class_names = plots.load_class_names()

    # TODO 2: Build the MobileNetV2 classifier.
    # ----------------------------------------------------------------------
    # Follow the Functional API pattern in EfficientNetV2-B0/model.py, adapting
    # its architecture to this model:
    #   raw RGB input (INPUT_SIZE, INPUT_SIZE, 3), values 0..255
    #   -> Rescaling(1 / 127.5, offset=-1)
    #   -> MobileNetV2(include_top=False, weights="imagenet", alpha=ALPHA)
    #   -> GlobalAveragePooling2D
    #   -> Dropout(DROPOUT)
    #   -> Dense(NUM_CLASSES, activation="softmax")
    # MobileNetV2 needs the explicit rescaling to [-1, 1]. EfficientNetV2 has
    # different built-in preprocessing, so do not copy that part unchanged.
    # Call the backbone with training=False to keep BatchNorm statistics fixed.
    # At 32x32 the output is 1x1x1280; at 64x64 it is 2x2x1280. Pool once.
    # Syntax hints:
    #   inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    #   x = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    #   backbone = keras.applications.MobileNetV2(
    #       input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False,
    #       weights="imagenet", alpha=ALPHA)
    #   x = backbone(x, training=False)  # call a layer by adding (x)
    #   x = keras.layers.GlobalAveragePooling2D()(x)
    #   x = keras.layers.Dropout(DROPOUT)(x)
    #   outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    #   model = keras.Model(inputs, outputs)
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=ALPHA
    )
    base_model.trainable = False  # Freeze the base model
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # TODO 3: Compile the model for integer Fashion-MNIST labels.
    # ----------------------------------------------------------------------
    # Use Adam(HEAD_LEARNING_RATE), sparse categorical cross-entropy, and
    # sparse categorical accuracy. Sparse loss matches the integer labels;
    # one-hot encoding is not needed.
    # Syntax hint: model.compile(optimizer=..., loss=..., metrics=[...])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HEAD_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    # TODO 4: Stage 1 - train only the new classification head.
    # ----------------------------------------------------------------------
    # Keep the MobileNetV2 backbone frozen. Fit on training data and pass
    # validation data to model.fit so you can monitor generalization.
    # Add EpochTimer, ModelCheckpoint(monitor="val_loss", save_best_only=True),
    # and EarlyStopping(restore_best_weights=True). These select and preserve
    # the best stage-1 weights using validation results.
    # Save stage-1 history and optionally report its validation accuracy.
    # For the two-phase pattern, see the Phase 1 section of EfficientNet's file.
    # Syntax hint: history = model.fit(train_ds, epochs=HEAD_EPOCHS,
    #                                 validation_data=val_ds, callbacks=[...])
    print("\nStage 1: training the classifier head")
    model.summary(show_trainable=True)

    head_timer = EpochTimer()
    head_history = model.fit(
        train_ds,
        epochs=HEAD_EPOCHS,
        validation_data=val_ds,
        callbacks=[
            head_timer,
            keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / "mobilenetv2_head_best.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=HEAD_PATIENCE,
                restore_best_weights=True,
            ),
        ],
    )

    # Explicitly restore the saved best head BEFORE changing trainability.
    # load_weights keeps base_model attached to this same model object.
    head_checkpoint = OUTPUT_DIR / "mobilenetv2_head_best.keras"
    model.load_weights(str(head_checkpoint))
    plots.save_history(head_history, str(OUTPUT_DIR / "stage1_history.json"))
    head_metrics, _, _ = save_classification_report(
        model, val_ds, class_names, OUTPUT_DIR / "stage1_validation", "validation",
    )
    save_json({"epoch_times_s": head_timer.epoch_times, "median_epoch_time_s": head_timer.median},
              OUTPUT_DIR / "stage1_timings.json")
    print(f"Best Stage 1 validation loss: {min(head_history.history['val_loss']):.4f}")
    # TODO 5: Stage 2 - fine-tune part of the pretrained backbone.
    # ----------------------------------------------------------------------
    # Set the backbone trainable, then calculate the target from ALL layers:
    #   n_unfreeze = int(len(base_model.layers) * FINE_TUNE_FRACTION)
    # Fill that quota from the end, SKIPPING every BatchNormalization layer.
    # MobileNet: int(154 * 0.3) = 46 non-BN layers marked trainable.
    # Some layers (ReLU/Add/etc.) have no weights; report weight-bearing layers too.
    # Recompile after changing trainability, using the smaller fine-tuning rate.
    # Stage 2 continues from the best stage-1 weights, not from scratch.
    # Syntax hint: eligible_layers = [layer for layer in base_model.layers
    #                               if not isinstance(layer, keras.layers.BatchNormalization)]
    if not np.isfinite(FINE_TUNE_FRACTION) or not 0 <= FINE_TUNE_FRACTION <= 1:
        raise ValueError("FINE_TUNE_FRACTION must be finite and between 0 and 1")
    base_model.trainable = True  # Allow fine-tuning to start
    # Freeze EVERY layer first. This also makes fraction=0 safe (no [:-0] bug).
    for layer in base_model.layers:
        layer.trainable = False
    n_unfreeze = int(len(base_model.layers) * FINE_TUNE_FRACTION)
    eligible_layers = [layer for layer in base_model.layers
                       if not isinstance(layer, keras.layers.BatchNormalization)]
    selected_layers = eligible_layers[-n_unfreeze:] if n_unfreeze else []
    for layer in selected_layers:
        layer.trainable = True
    # If requested layers exceed the available non-BN count, use all eligible layers.
    if len(selected_layers) < n_unfreeze:
        print("Requested quota exceeds non-BN layers; selecting all eligible layers.")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    trainable_backbone_layers = sum(layer.trainable for layer in base_model.layers)
    print(
        f"\nStage 2: requested {n_unfreeze} of {len(base_model.layers)} layers; "
        f"{trainable_backbone_layers} non-BN layers are trainable "
        f"({sum(bool(layer.trainable_weights) for layer in base_model.layers)} have trainable weights)."
    )
    config["fine_tuning"] = {
        "rule": "fraction_of_total_layers_skipping_batchnorm",
        "backbone_layers": len(base_model.layers), "requested_layers": n_unfreeze,
        "eligible_non_batchnorm_layers": len(eligible_layers),
        "trainable_backbone_layers": trainable_backbone_layers,
        "trainable_weight_layers": sum(bool(layer.trainable_weights) for layer in base_model.layers),
        "selected_layer_names": [layer.name for layer in selected_layers],
    }
    save_json(config, OUTPUT_DIR / "config.json")
    model.summary(show_trainable=True)

    fine_tune_timer = EpochTimer()
    fine_tune_history = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=[
            fine_tune_timer,
            keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / "mobilenetv2_finetuned_best.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=FINE_TUNE_PATIENCE,
                restore_best_weights=True,
            ),
        ],
    )

    fine_checkpoint = OUTPUT_DIR / "mobilenetv2_finetuned_best.keras"
    model.load_weights(str(fine_checkpoint))
    plots.save_history(fine_tune_history, str(OUTPUT_DIR / "stage2_history.json"))
    fine_metrics, _, _ = save_classification_report(
        model, val_ds, class_names, OUTPUT_DIR / "stage2_validation", "validation",
    )
    save_json({"epoch_times_s": fine_tune_timer.epoch_times,
               "median_epoch_time_s": fine_tune_timer.median}, OUTPUT_DIR / "stage2_timings.json")
    print(
        "Best Stage 2 validation loss: "
        f"{min(fine_tune_history.history['val_loss']):.4f}"
    )

    # TODO 6: Save the selected model; final TEST evaluation is a separate mode.
    # ----------------------------------------------------------------------
    # Compare both stages using VALIDATION loss, never test accuracy.
    # Fine-tuning can get worse, so allow the best head to remain the winner.
    # Syntax hint: keras.models.load_model(checkpoint, compile=False)
    # Your original test prediction loop remains in evaluate_saved_run() below.
    # Run this same file with --test_with_test_data RUN_DIR when choices are final.
    if not all(np.isfinite(item["validation_loss"]) for item in (head_metrics, fine_metrics)):
        raise ValueError("Cannot select a model with non-finite validation loss")
    if head_metrics["validation_loss"] <= fine_metrics["validation_loss"]:
        selected_checkpoint, selected_metrics = head_checkpoint, head_metrics
        selected_stage, selected_history = "head", head_history
    else:
        selected_checkpoint, selected_metrics = fine_checkpoint, fine_metrics
        selected_stage, selected_history = "fine_tuning", fine_tune_history
    model = keras.models.load_model(str(selected_checkpoint), compile=False)
    inference_path = OUTPUT_DIR / "mobilenetv2_selected_inference.keras"
    model.save(str(inference_path))  # compile=False excludes Adam's optimizer state.

    # These measurements use the selected model and synthetic input, not test data.
    # The existing evaluation helpers distinguish checkpoint and inference sizes.
    # estimate_macs still uses FLOPs/2; count_macs() has NOT been changed.
    print("\nMeasuring model computation and CPU inference latency")
    measurements = model_reporting_metrics(model, selected_checkpoint, inference_path)
    save_json(measurements, OUTPUT_DIR / "model_metrics.json")
    config.update({
        "selected_stage": selected_stage,
        "selected_epoch": int(np.argmin(selected_history.history["val_loss"])) + 1,
        "selected_training_checkpoint": selected_checkpoint.name,
        "selected_model_file": inference_path.name,
        "selected_validation_loss": selected_metrics["validation_loss"],
        "selected_validation_accuracy": selected_metrics["validation_accuracy"],
    })
    save_json(config, OUTPUT_DIR / "config.json")
    # TODO 7: Save MobileNetV2's report files separately from EfficientNet's.
    # ----------------------------------------------------------------------
    # Use OUTPUT_DIR for metrics JSON/CSV, training histories, confusion
    # matrix, figures, and checkpoints. Reuse src/plots.py where suitable.
    # Do not save MobileNetV2 figures in results/figures or the EfficientNet
    # output folder; the separate results/mobilenetv2/ path avoids overwrites.
    # Syntax hint: report_path = OUTPUT_DIR / "metrics.json"

    # Plot training/validation histories and epoch times; no test predictions.
    plots.plot_learning_curves(
        head_history, str(OUTPUT_DIR / "figures" / "stage1"), "Stage 1"
    )
    plots.plot_learning_curves(
        fine_tune_history, str(OUTPUT_DIR / "figures" / "stage2"), "Stage 2"
    )
    plots.plot_full_training_curve(
        head_history, fine_tune_history, str(OUTPUT_DIR / "figures" / "comparison")
    )
    plots.plot_epoch_times(
        head_timer.epoch_times, str(OUTPUT_DIR / "figures" / "stage1"), "Stage 1 epoch times"
    )
    plots.plot_epoch_times(
        fine_tune_timer.epoch_times,
        str(OUTPUT_DIR / "figures" / "stage2"),
        "Stage 2 epoch times",
    )

    metrics = {**selected_metrics, **measurements, "selected_stage": selected_stage,
               "input_size": INPUT_SIZE, "fine_tune_fraction": FINE_TUNE_FRACTION}
    save_json(metrics, OUTPUT_DIR / "metrics.json")
    print("\nSelected model: ", selected_stage)
    print(json.dumps(metrics, indent=2))
    print("Test data has not been evaluated. After choosing the configuration, run:")
    print(f'python MobileNetV2/model.py --test_with_test_data "{OUTPUT_DIR}"')


def evaluate_saved_run(run_dir):
    """Final test only: reuse the selected checkpoint without fitting any model.

    Example: python MobileNetV2/model.py --test_with_test_data "path/to/run_folder"
    Read settings from the saved run, so later edits to constants cannot change it.
    """
    import keras
    from src import plots
    from src.data_loader import DATA_FILE, make_tf_datasets
    from src.evaluation import dataset_digest
    from src.utils import save_json

    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["model_name"] != "MobileNetV2":
        raise ValueError("Use a MobileNetV2 run directory with this script")
    if dataset_digest(DATA_FILE) != config["dataset_sha256"]:
        raise ValueError("The processed splits differ from this training run")
    model = keras.models.load_model(str(run_dir / config["selected_model_file"]), compile=False)
    if model.input_shape[1:] != (config["input_size"], config["input_size"], 3):
        raise ValueError("Checkpoint shape disagrees with the recorded configuration")
    # Reuse the original loader. Only its test dataset is evaluated in this mode.
    _, _, test_ds = make_tf_datasets(
        batch_size=config["batch_size"], sota=True, size=config["input_size"],
        shuffle_seed=config["seed"],
    )
    # Keep every explicitly requested test report separate too.
    test_base = run_dir / "final_test"
    test_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%SZ_")
    OUTPUT_DIR = Path(tempfile.mkdtemp(prefix=stamp, dir=str(test_base)))
    NUM_CLASSES = config["num_classes"]
    print("\nFinal evaluation on the test set")

    # Collect labels and predictions in one pass through the test dataset.
    true_batches = []
    probability_batches = []

    for images, labels in test_ds:
        probabilities = model(images, training=False).numpy()
        true_batches.append(labels.numpy())
        probability_batches.append(probabilities)

    y_true = np.concatenate(true_batches).astype(int)
    y_prob = np.concatenate(probability_batches)
    y_pred = np.argmax(y_prob, axis=1)

    test_loss = float(
        keras.losses.SparseCategoricalCrossentropy()(y_true, y_prob).numpy()
    )
    test_accuracy = float(np.mean(y_true == y_pred))

    class_names = plots.load_class_names()
    precision, recall, f1 = plots.per_class_scores(y_true, y_pred, NUM_CLASSES)
    confusion_matrix = plots.plot_confusion_matrix(
        y_true, y_pred, class_names, str(OUTPUT_DIR / "figures" / "final"),
        "MobileNetV2 test confusion matrix",
    )
    np.savetxt(
        OUTPUT_DIR / "confusion_matrix.csv",
        confusion_matrix,
        delimiter=",",
        fmt="%d",
    )
    plots.plot_per_class_metrics(
        y_true, y_pred, class_names, str(OUTPUT_DIR / "figures" / "final"),
        "MobileNetV2 test metrics by class",
    )

    measurements = json.loads((run_dir / "model_metrics.json").read_text(encoding="utf-8"))
    metrics = {
        **measurements, "split": "test", "model_name": config["model_name"],
        "input_size": config["input_size"],
        "fine_tune_fraction": config["fine_tune_fraction"],
        "selected_stage": config["selected_stage"], "selected_epoch": config["selected_epoch"],
        "dataset_sha256": config["dataset_sha256"], "test_images": len(y_true),
        "test_loss": test_loss, "test_accuracy": test_accuracy,
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }
    save_json(metrics, OUTPUT_DIR / "metrics.json")
    class_report = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(NUM_CLASSES)
    }
    with open(OUTPUT_DIR / "classification_report.json", "w") as report_file:
        json.dump(class_report, report_file, indent=2)

    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Test reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
