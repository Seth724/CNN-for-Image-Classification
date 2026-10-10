from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "fashion_mnist_70_15_15.npz"


def _raw():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Shared processed dataset not found: {DATA_FILE}\n"
            "Copy the team's processed file or run scripts/prepare_data.py once."
        )
    d = np.load(DATA_FILE)
    return d["x_train"], d["y_train"], d["x_val"], d["y_val"], d["x_test"], d["y_test"]


def load_custom_cnn_data():
    x_train, y_train, x_val, y_val, x_test, y_test = _raw()
    x_train = x_train.astype(np.float32)[..., None] / 255.0
    x_val = x_val.astype(np.float32)[..., None] / 255.0
    x_test = x_test.astype(np.float32)[..., None] / 255.0
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_sota_data(size=32):
    if size < 32:
        raise ValueError("Use size >= 32 for the pretrained-model pipeline")
    x_train, y_train, x_val, y_val, x_test, y_test = _raw()

    def convert(x):
        x = tf.convert_to_tensor(x[..., None], dtype=tf.float32)
        x = tf.image.resize(x, (size, size))
        x = tf.image.grayscale_to_rgb(x)
        return x.numpy()

    return convert(x_train), y_train, convert(x_val), y_val, convert(x_test), y_test


def make_tf_datasets(batch_size=128, sota=False, size=32, shuffle_seed=42):
    if sota:
        x_train, y_train, x_val, y_val, x_test, y_test = load_sota_data(size=size)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = load_custom_cnn_data()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train), seed=shuffle_seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, val_ds, test_ds
