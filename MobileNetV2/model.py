import json
import os
from pathlib import Path
import sys
import numpy as np
# Make project imports work when this file is run from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import keras

from src import plots
from src.data_loader import make_tf_datasets
from src.evaluation import (
    EpochTimer,
    estimate_flops,
    estimate_macs,
    measure_cpu_latency_ms,
    model_size_mb,
)


# Keep these choices together so the experiment setup is easy to review.
INPUT_SIZE = 32            # Match the current EfficientNet input for Q6 comparison.
NUM_CLASSES = 10           # Fashion-MNIST has ten labels.
BATCH_SIZE = 128
SEED = 42
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_FRACTION = 0.3   # Fine-tune the last 30%, like the EfficientNet script.
ALPHA = 1.0                # Keep the standard MobileNetV2 width for the baseline.

# Each SOTA model must keep its own figures, metrics, logs, and checkpoints.
# Keep the normal local destination, but allow Colab to point outputs to Drive.
# Example: set MOBILENETV2_OUTPUT_DIR before launching this script in Colab.
OUTPUT_DIR = Path(
    os.environ.get(
        "MOBILENETV2_OUTPUT_DIR",
        str(ROOT / "results" / "mobilenetv2"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A fixed seed makes data shuffling and Keras randomness more repeatable.
keras.utils.set_random_seed(SEED)


def main():
    # ----------------------------------------------------------------------
    # TODO 1: Load the shared train, validation, and test splits.
    # ----------------------------------------------------------------------
    # Use make_tf_datasets(batch_size=BATCH_SIZE, sota=True, size=INPUT_SIZE).
    # The SOTA loader resizes each batch and converts grayscale images to RGB.
    # Its image values are still in the 0..255 range at this point.
    # Do not create a new split and do not resize the full dataset in memory.
    train_ds, val_ds, test_ds = make_tf_datasets(
        batch_size=BATCH_SIZE,
        sota=True,
        size=INPUT_SIZE,
    )
    # Use the three shared datasets returned by the loader as-is.

    # TODO 2: Build the MobileNetV2 classifier.
    # ----------------------------------------------------------------------
    # Follow the Functional API pattern in EfficientNetV2-B0/model.py, adapting
    # its architecture to this model:
    #   raw RGB input (32, 32, 3), values 0..255
    #   -> Rescaling(1 / 127.5, offset=-1)
    #   -> MobileNetV2(include_top=False, weights="imagenet", alpha=ALPHA)
    #   -> GlobalAveragePooling2D
    #   -> Dropout(0.2)
    #   -> Dense(NUM_CLASSES, activation="softmax")
    # MobileNetV2 needs the explicit rescaling to [-1, 1]. EfficientNetV2 has
    # different built-in preprocessing, so do not copy that part unchanged.
    # Call the backbone with training=False to keep BatchNorm statistics fixed.
    # At 32x32, the backbone output is about 1x1x1280; use pooling only once.
    # Syntax hints:
    #   inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    #   x = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    #   backbone = keras.applications.MobileNetV2(
    #       input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False,
    #       weights="imagenet", alpha=ALPHA)
    #   x = backbone(x, training=False)  # call a layer by adding (x)
    #   x = keras.layers.GlobalAveragePooling2D()(x)
    #   x = keras.layers.Dropout(0.2)(x)
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
    x = keras.layers.Dropout(0.2)(x)
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
                patience=3,
                restore_best_weights=True,
            ),
        ],
    )

    plots.save_history(head_history, str(OUTPUT_DIR / "stage1_history.json"))
    print(f"Best Stage 1 validation loss: {min(head_history.history['val_loss']):.4f}")
    # TODO 5: Stage 2 - fine-tune part of the pretrained backbone.
    # ----------------------------------------------------------------------
    # Set the backbone trainable, then calculate how many layers to unfreeze:
    #   n_unfreeze = int(len(backbone.layers) * FINE_TUNE_FRACTION)
    # Freeze all layers before the last n_unfreeze layers, like the teammate's
    # EfficientNet script. Keep every BatchNormalization layer frozen. The
    # exact count comes from the loaded MobileNetV2 backbone, so print it when
    # you implement this section. Recompile after changing trainability, with
    # Adam(FINE_TUNE_LEARNING_RATE): a small rate reduces the risk of
    # overwriting useful pretrained features too quickly.
    # Fit again on training data with validation monitoring. Add a new timer,
    # checkpoint, EarlyStopping, and optionally ReduceLROnPlateau.
    # Stage 2 continues from stage-1 weights; it is not training from scratch.
    # See the Phase 2 section of EfficientNet's file for the training pattern.
    # Syntax hint:
    #   for layer in backbone.layers[:-n_unfreeze]:
    #       layer.trainable = False
    base_model.trainable = True  # Allow fine-tuning to start

    n_unfreeze = int(len(base_model.layers) * FINE_TUNE_FRACTION)
    for layer in base_model.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    trainable_backbone_layers = sum(layer.trainable for layer in base_model.layers)
    print(
        f"\nStage 2: selected the last {n_unfreeze} of {len(base_model.layers)} "
        f"layers; {trainable_backbone_layers} remain trainable after freezing BatchNorm."
    )
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
                patience=5,
                restore_best_weights=True,
            ),
        ],
    )

    plots.save_history(fine_tune_history, str(OUTPUT_DIR / "stage2_history.json"))
    print(
        "Best Stage 2 validation loss: "
        f"{min(fine_tune_history.history['val_loss']):.4f}"
    )

    # These measurements use the trained model and synthetic input, not test data.
    # FLOPs are profiled once; estimate_macs reuses that count as an approximation.
    print("\nMeasuring model computation and CPU inference latency")
    flops_per_image = estimate_flops(model)
    macs_per_image = estimate_macs(model, flops=flops_per_image)
    try:
        cpu_latency_median_ms = measure_cpu_latency_ms(model)
    except Exception as error:
        # CPU timing is optional; keep final test evaluation usable if this
        # TensorFlow/Keras version cannot clone or trace the model on CPU.
        print(f"CPU latency measurement unavailable: {error}")
        cpu_latency_median_ms = None
    print(f"Estimated FLOPs/image: {flops_per_image}")
    print(f"Estimated MACs/image: {macs_per_image}")
    print(f"Median CPU latency, batch size 1 (ms/image): {cpu_latency_median_ms}")

    # TODO 6: Save the final model and record final test results.
    # ----------------------------------------------------------------------
    # Save the fine-tuned model under OUTPUT_DIR. Use the final test split only
    # after all choices are complete. Make one prediction pass over test data,
    # then calculate accuracy, confusion matrix, precision, and recall from
    # those predictions. Do not use test scores to change training settings.
    # The assignment asks for total parameters and model size in MB too.
    # For Q6, record FLOPs/MACs, CPU latency, accuracy, and model size for
    # comparison against Model B. These measurements do not use test images.
    # Syntax hints:
    #   model.save(str(OUTPUT_DIR / "mobilenetv2_finetuned.keras"))
    #   probabilities = model.predict(test_ds)
    print("\nFinal evaluation on the test set")
    model.save(str(OUTPUT_DIR / "mobilenetv2_finetuned.keras"))

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
    # TODO 7: Save MobileNetV2's report files separately from EfficientNet's.
    # ----------------------------------------------------------------------
    # Use OUTPUT_DIR for metrics JSON/CSV, training histories, confusion
    # matrix, figures, and checkpoints. Reuse src/plots.py where suitable.
    # Do not save MobileNetV2 figures in results/figures or the EfficientNet
    # output folder; the separate results/mobilenetv2/ path avoids overwrites.
    # Syntax hint: report_path = OUTPUT_DIR / "metrics.json"

    plots.plot_per_class_metrics(
        y_true, y_pred, class_names, str(OUTPUT_DIR / "figures" / "final"),
        "MobileNetV2 test metrics by class",
    )

    # Plot training curves and epoch times without running test predictions again.
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

    metrics = {
        "input_size": INPUT_SIZE,
        "fine_tune_fraction": FINE_TUNE_FRACTION,
        "backbone_layers": len(base_model.layers),
        "selected_for_fine_tuning": n_unfreeze,
        "trainable_backbone_layers": int(trainable_backbone_layers),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(
            sum(np.prod(weight.shape) for weight in model.trainable_weights)
        ),
        "model_size_mb": model_size_mb(model),
        "flops_per_image": flops_per_image,
        "estimated_macs_per_image": macs_per_image,
        "cpu_inference_median_ms_per_image_batch1": cpu_latency_median_ms,
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as report_file:
        json.dump(metrics, report_file, indent=2)

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
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
