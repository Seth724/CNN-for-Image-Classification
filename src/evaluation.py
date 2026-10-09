"""Model-agnostic metrics shared by the custom CNN and the SOTA model.

Keeping this outside the model scripts guarantees both models are measured
with exactly the same code, which is what makes the comparison fair.
"""
import json
import os
import tempfile
import time

import keras
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
    """Size of the model saved to disk (.keras) in MB."""
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


def predict_labels(model, ds):
    """Return (y_true, y_pred) for an unshuffled dataset such as test_ds."""
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)
    return y_true, y_pred


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
