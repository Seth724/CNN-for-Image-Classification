import argparse
from datetime import datetime, timezone
import json
import os
import sys
import pathlib
import tempfile
import numpy as np

# Make the repo root importable so `src` resolves when running this file directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Match MobileNet's 64x64 input for the resolution comparison.
# Keep the teammate's batch size, epoch limits and fine-tuning patience.
INPUT_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 10
SEED = 42
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
HEAD_PATIENCE = 3
FINE_TUNE_PATIENCE = 4
UNFREEZE_FRACTION = 0.3   # Target 30% of all layers; skip BN while selecting.
DROPOUT = 0.2
OUTPUT_BASE = pathlib.Path(os.environ.get(
    "EFFICIENTNETV2_OUTPUT_DIR", str(ROOT / "results" / "efficientnetv2_b0_64"),
))


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-B0 using validation data.")
    parser.add_argument("--test_with_test_data", type=pathlib.Path, metavar="RUN_DIR",
                        help="Evaluate an already selected run on test data; no training")
    args = parser.parse_args()
    if args.test_with_test_data is not None:
        evaluate_saved_run(args.test_with_test_data)
        return

    # Import here so --help and importing this file do not require TensorFlow.
    import keras
    import tensorflow as tf
    from src.data_loader import DATA_FILE, make_tf_datasets
    from src.evaluation import EpochTimer, dataset_digest, model_reporting_metrics, save_classification_report
    from src import plots
    from src.utils import save_json

    # Folder creation and seeding happen only when running, never on import.
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%SZ_")
    OUT_DIR = pathlib.Path(tempfile.mkdtemp(prefix=stamp, dir=str(OUTPUT_BASE)))
    keras.utils.set_random_seed(SEED)
    print(f"Run directory: {OUT_DIR}")
    config = {
        "model_name": "EfficientNetV2-B0", "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES, "batch_size": BATCH_SIZE, "seed": SEED,
        "head_epochs": HEAD_EPOCHS, "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
        "head_patience": HEAD_PATIENCE, "fine_tune_patience": FINE_TUNE_PATIENCE,
        "fine_tune_fraction": UNFREEZE_FRACTION, "dropout": DROPOUT,
        "batchnorm_frozen": True, "augmentation": False,
        "selection_metric": "validation_loss", "preprocessing": "EfficientNetV2 built-in",
        "dataset_sha256": dataset_digest(DATA_FILE),
        "keras_version": keras.__version__, "tensorflow_version": tf.__version__,
    }
    save_json(config, OUT_DIR / "config.json")

    # The original loader returns all three splits; discard the test dataset here.
    # Training and stage reports use only train_ds / val_ds. Final testing is below.
    train_ds, val_ds, _ = make_tf_datasets(
        batch_size=BATCH_SIZE, sota=True, size=INPUT_SIZE, shuffle_seed=SEED,
    )

    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_preprocessing=True,  # Built-in scaling expects raw RGB pixels in 0..255.
    )

    base_model.trainable = False  # Freeze the base model

    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

    # Keep BatchNorm statistics fixed; selected convolution weights can still learn.
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.summary(show_trainable=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HEAD_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # ---------------- Phase 1: transfer learning (backbone frozen) ----------------
    epochs = HEAD_EPOCHS
    print("Phase 1: training the head for {} epochs...".format(epochs))
    timer = EpochTimer()
    history_transfer = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[
            timer,
            keras.callbacks.ModelCheckpoint(str(OUT_DIR / "transfer_best.keras"),
                                            monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(str(OUT_DIR / "transfer_training.csv")),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=HEAD_PATIENCE, restore_best_weights=True),
        ],
    )

    # Restore the BEST head weights before starting fine-tuning.
    # load_weights preserves the base_model object used in Phase 2 below.
    head_checkpoint = OUT_DIR / "transfer_best.keras"
    model.load_weights(str(head_checkpoint))
    model.save(str(OUT_DIR / "transfer.keras"))  # lets phase 2 be rerun on its own

    print("\nPhase 1 metrics on the validation set:")
    FIG_DIR = OUT_DIR / "figures"
    class_names = plots.load_class_names()
    metrics_transfer, _, _, y_true, pred_transfer = save_classification_report(
        model, val_ds, class_names, OUT_DIR / "transfer_validation", "validation",
        return_predictions=True,
    )
    metrics_transfer.update({
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "training_checkpoint_size_mib": head_checkpoint.stat().st_size / 1024**2,
        "median_epoch_time_s": timer.median,
    })
    save_json(metrics_transfer, OUT_DIR / "transfer_metrics.json")
    save_json({"epoch_times_s": timer.epoch_times, "median_epoch_time_s": timer.median},
              OUT_DIR / "transfer_timings.json")
    plots.save_history(history_transfer, str(OUT_DIR / "transfer_history.json"))
    # Reuse the validation predictions above; these plots do not predict again.
    plots.plot_learning_curves(history_transfer, f"{FIG_DIR}/transfer", "EfficientNetV2-B0 transfer learning")
    plots.plot_epoch_times(timer.epoch_times, f"{FIG_DIR}/transfer", "Transfer learning: epoch times")


    # ---------------- Phase 2: fine-tuning (last part of backbone unfrozen) ----------------
    # UNFREEZE_FRACTION is set with the other experiment settings at the top.
    # Target a fraction of ALL backbone layers, but skip BN when filling the quota.
    # Freeze every layer first, so even fraction=0 behaves correctly.
    if not np.isfinite(UNFREEZE_FRACTION) or not 0 <= UNFREEZE_FRACTION <= 1:
        raise ValueError("UNFREEZE_FRACTION must be finite and between 0 and 1")
    fine_tune_epochs = FINE_TUNE_EPOCHS
    fine_tune_lr = FINE_TUNE_LEARNING_RATE
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    n_unfreeze = int(len(base_model.layers) * UNFREEZE_FRACTION)
    eligible_layers = [layer for layer in base_model.layers
                       if not isinstance(layer, keras.layers.BatchNormalization)]
    selected_layers = eligible_layers[-n_unfreeze:] if n_unfreeze else []
    for layer in selected_layers:
        layer.trainable = True
    # Every BN remains frozen, and BN layers do not consume the requested quota.
    trainable_backbone_layers = sum(layer.trainable for layer in base_model.layers)
    print("Phase 2: requested {} of {} backbone layers; {} non-BN layers trainable ({} with weights)".format(
        n_unfreeze, len(base_model.layers), trainable_backbone_layers,
        sum(bool(layer.trainable_weights) for layer in base_model.layers),
    ))
    if len(selected_layers) < n_unfreeze:
        print("Requested quota exceeds non-BN layers; selecting all eligible layers.")
    config["fine_tuning"] = {
        "rule": "fraction_of_total_layers_skipping_batchnorm",
        "backbone_layers": len(base_model.layers), "requested_layers": n_unfreeze,
        "eligible_non_batchnorm_layers": len(eligible_layers),
        "trainable_backbone_layers": trainable_backbone_layers,
        "trainable_weight_layers": sum(bool(layer.trainable_weights) for layer in base_model.layers),
        "selected_layer_names": [layer.name for layer in selected_layers],
    }
    save_json(config, OUT_DIR / "config.json")

    # Recompile after changing trainability, with a fresh, smaller-rate optimizer.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary(show_trainable=True)

    ft_timer = EpochTimer()
    history_finetune = model.fit(
        train_ds,
        epochs=fine_tune_epochs,
        validation_data=val_ds,
        callbacks=[
            ft_timer,
            keras.callbacks.ModelCheckpoint(str(OUT_DIR / "finetuned_best.keras"),
                                            monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(str(OUT_DIR / "finetuned_training.csv")),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=FINE_TUNE_PATIENCE, restore_best_weights=True),
        ],
    )

    fine_checkpoint = OUT_DIR / "finetuned_best.keras"
    model.load_weights(str(fine_checkpoint))
    model.save(str(OUT_DIR / "finetuned.keras"))

    print("\nPhase 2 (fine-tuned) metrics on the validation set:")
    metrics_finetuned, _, _, fine_y_true, pred_finetuned = save_classification_report(
        model, val_ds, class_names, OUT_DIR / "finetuned_validation", "validation",
        return_predictions=True,
    )
    if not np.array_equal(y_true, fine_y_true):
        raise ValueError("Validation label order changed between stages")
    metrics_finetuned.update({
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "training_checkpoint_size_mib": fine_checkpoint.stat().st_size / 1024**2,
        "median_epoch_time_s": ft_timer.median,
    })
    save_json(metrics_finetuned, OUT_DIR / "finetuned_metrics.json")
    save_json({"epoch_times_s": ft_timer.epoch_times, "median_epoch_time_s": ft_timer.median},
              OUT_DIR / "finetuned_timings.json")

    # ---------------- Fine tuned model plots ----------------
    # Reports above use VALIDATION predictions, rather than the test set.
    plots.save_history(history_finetune, str(OUT_DIR / "finetuned_history.json"))
    plots.plot_learning_curves(history_finetune, f"{FIG_DIR}/finetuned", "EfficientNetV2-B0 fine-tuning")
    plots.plot_epoch_times(ft_timer.epoch_times, f"{FIG_DIR}/finetuned", "Fine-tuning: epoch times")
    plots.plot_full_training_curve(history_transfer, history_finetune, f"{FIG_DIR}/comparison")
    plots.plot_metric_comparison(metrics_transfer, metrics_finetuned, f"{FIG_DIR}/comparison", split="validation")
    plots.plot_per_class_comparison(y_true, pred_transfer, pred_finetuned, class_names, f"{FIG_DIR}/comparison")
    plots.plot_efficiency_table(metrics_transfer, metrics_finetuned, f"{FIG_DIR}/comparison", split="validation")

    # ---------------- Select and export using validation loss only ----------------
    if not all(np.isfinite(item["validation_loss"]) for item in (metrics_transfer, metrics_finetuned)):
        raise ValueError("Cannot select a model with non-finite validation loss")
    if metrics_transfer["validation_loss"] <= metrics_finetuned["validation_loss"]:
        selected_checkpoint, selected_metrics = head_checkpoint, metrics_transfer
        selected_stage, selected_history = "head", history_transfer
    else:
        selected_checkpoint, selected_metrics = fine_checkpoint, metrics_finetuned
        selected_stage, selected_history = "fine_tuning", history_finetune
    # Export without Adam state; this size differs from a training checkpoint's size.
    model = keras.models.load_model(str(selected_checkpoint), compile=False)
    inference_path = OUT_DIR / "efficientnetv2_b0_selected_inference.keras"
    model.save(str(inference_path))
    config.update({
        "selected_stage": selected_stage,
        "selected_epoch": int(np.argmin(selected_history.history["val_loss"])) + 1,
        "selected_training_checkpoint": selected_checkpoint.name,
        "selected_model_file": inference_path.name,
        "selected_validation_loss": selected_metrics["validation_loss"],
        "selected_validation_accuracy": selected_metrics["validation_accuracy"],
    })
    save_json(config, OUT_DIR / "config.json")
    # FLOPs/MACs and CPU timing use the model and synthetic input, not test images.
    print("\nMeasuring model computation and CPU inference latency")
    measurements = model_reporting_metrics(model, selected_checkpoint, inference_path)
    save_json(measurements, OUT_DIR / "model_metrics.json")
    save_json({**selected_metrics, **measurements, "selected_stage": selected_stage}, OUT_DIR / "metrics.json")
    print(json.dumps({**selected_metrics, **measurements}, indent=2))
    print("Test data has not been evaluated. After choosing the configuration, run:")
    print(f'python EfficientNetV2-B0/model.py --test_with_test_data "{OUT_DIR}"')


def evaluate_saved_run(run_dir):
    """Final test only: load the selected model, with no compile/fit or tuning.

    Example: python EfficientNetV2-B0/model.py --test_with_test_data "path/to/run_folder"
    """
    import keras
    from src import plots
    from src.data_loader import DATA_FILE, make_tf_datasets
    from src.evaluation import dataset_digest, save_classification_report
    from src.utils import save_json

    run_dir = pathlib.Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["model_name"] != "EfficientNetV2-B0":
        raise ValueError("Use an EfficientNetV2-B0 run directory with this script")
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
    test_base = run_dir / "final_test"
    test_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%SZ_")
    out = pathlib.Path(tempfile.mkdtemp(prefix=stamp, dir=str(test_base)))
    # The existing helper predicts once and saves loss, accuracy, P/R/F1 and plots.
    metrics, _, _ = save_classification_report(model, test_ds, plots.load_class_names(), out, "test")
    measurements = json.loads((run_dir / "model_metrics.json").read_text(encoding="utf-8"))
    metrics.update({**measurements, "model_name": config["model_name"],
                    "input_size": config["input_size"], "fine_tune_fraction": config["fine_tune_fraction"],
                    "selected_stage": config["selected_stage"], "selected_epoch": config["selected_epoch"],
                    "dataset_sha256": config["dataset_sha256"]})
    save_json(metrics, out / "metrics.json")
    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Test reports saved to: {out}")


if __name__ == "__main__":
    main()
