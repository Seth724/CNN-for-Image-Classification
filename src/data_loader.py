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


def make_tf_datasets(batch_size=128, sota=False, size=32, shuffle_seed=42):
    if sota:
        return make_sota_datasets_streaming(batch_size, size, shuffle_seed)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = load_custom_cnn_data()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache().shuffle(len(x_train), seed=shuffle_seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, val_ds, test_ds


def make_sota_datasets_streaming(batch_size=32, size=32, shuffle_seed=42):
    """Resize raw images per batch, preserving 0–255 pixels for model preprocessing.

    Training reshuffles each epoch; validation and test keep their original order.
    Expanded float RGB images are never cached or materialised for the full split.
    """
    if size < 32:
        raise ValueError("Use size >= 32 for the pretrained-model pipeline")
    x_train, y_train, x_val, y_val, x_test, y_test = _raw()

    def convert(images, labels):
        images = tf.cast(images[..., None], tf.float32)
        images = tf.image.resize(images, (size, size))
        return tf.image.grayscale_to_rgb(images), labels

    def build(images, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(len(images), seed=shuffle_seed, reshuffle_each_iteration=True)
        return (ds.batch(batch_size)
                .map(convert, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    return build(x_train, y_train, True), build(x_val, y_val), build(x_test, y_test)
