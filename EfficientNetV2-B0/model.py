import sys
import pathlib

# Make the repo root importable so `src` resolves when running this file directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import keras
from keras import layers
from src.data_loader import make_tf_datasets
from src.evaluation import EpochTimer, collect_metrics, predict_labels
from src import plots

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "efficientnetv2_b0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(42)

train_ds, val_ds, test_ds = make_tf_datasets(batch_size=32, sota=True, size=32)

base_model = keras.applications.EfficientNetV2B0(
    include_top=False,
    weights='imagenet',
    input_shape=(32, 32, 3),
)

base_model.trainable = False # Freeze the base model

inputs = keras.Input(shape=(32, 32, 3))

x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model  = keras.Model(inputs, outputs)

model.summary(show_trainable=True)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ---------------- Phase 1: transfer learning (backbone frozen) ----------------
epochs = 10
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
    ],
)

print("\nPhase 1 metrics on the test set:")
metrics_transfer = collect_metrics(model, test_ds, timer, num_classes=10, save_path=str(OUT_DIR / "transfer_metrics.json"))
model.save(str(OUT_DIR / "transfer.keras"))  # lets phase 2 be rerun on its own

FIG_DIR = OUT_DIR / "figures"
class_names = plots.load_class_names()
y_true, pred_transfer = predict_labels(model, test_ds)
plots.save_history(history_transfer, str(OUT_DIR / "transfer_history.json"))
plots.plot_stage_report(model, test_ds, history_transfer, timer.epoch_times, class_names,
                        f"{FIG_DIR}/transfer", "EfficientNetV2-B0 transfer learning")



# ---------------- Phase 2: fine-tuning (last part of backbone unfrozen) ----------------
UNFREEZE_FRACTION = 0.3   # last 20-30% of backbone layers
FREEZE_BATCHNORM = True    # keep BN frozen; set False only if training is stable without it
fine_tune_epochs = 20
fine_tune_lr = 1e-5

base_model.trainable = True
n_unfreeze = int(len(base_model.layers) * UNFREEZE_FRACTION)
for layer in base_model.layers[:-n_unfreeze]:
    layer.trainable = False
if FREEZE_BATCHNORM:
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
print("Phase 2: unfroze last {} of {} backbone layers".format(n_unfreeze, len(base_model.layers)))

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
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ],
)

model.save(str(OUT_DIR / "finetuned.keras"))

print("\nPhase 2 (fine-tuned) metrics on the test set:")
metrics_finetuned = collect_metrics(model, test_ds, ft_timer, num_classes=10, save_path=str(OUT_DIR / "finetuned_metrics.json"))

# ---------------- Fine tuned model plots ----------------
_, pred_finetuned = predict_labels(model, test_ds)
plots.save_history(history_finetune, str(OUT_DIR / "finetuned_history.json"))
plots.plot_stage_report(model, test_ds, history_finetune, ft_timer.epoch_times, class_names,
                        f"{FIG_DIR}/finetuned", "EfficientNetV2-B0 fine-tuning")

plots.plot_full_training_curve(history_transfer, history_finetune, f"{FIG_DIR}/comparison")
plots.plot_metric_comparison(metrics_transfer, metrics_finetuned, f"{FIG_DIR}/comparison")
plots.plot_per_class_comparison(y_true, pred_transfer, pred_finetuned, class_names, f"{FIG_DIR}/comparison")
plots.plot_efficiency_table(metrics_transfer, metrics_finetuned, f"{FIG_DIR}/comparison")
