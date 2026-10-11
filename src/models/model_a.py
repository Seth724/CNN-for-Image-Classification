from tensorflow import keras
from tensorflow.keras import layers


MODEL_A_PARAMETER_LIMIT = 100_000
MODEL_A_EXPECTED_PARAMS = 101_578


def build_model_a(input_shape=(28, 28, 1), num_classes=10) -> keras.Model:
    """Build the standard-convolution CNN used for Model A."""
    inputs = keras.Input(shape=input_shape, name="image")

    # 3x3 convolution with 32 filters, followed by 2x2 downsampling.
    x = layers.Conv2D(32, 3, strides=1, padding="same", activation="relu", name="conv1_relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2, name="pool1")(x)

    # 3x3 convolution with 64 filters, followed by 2x2 downsampling.
    x = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu", name="conv2_relu")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool2")(x)

    # 3x3 convolution with 128 filters at the 7x7 spatial resolution.
    x = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu", name="conv3_relu")(x)

    # Global average pooling keeps the classifier small and avoids Flatten.
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    # Compact fully connected classifier.
    x = layers.Dense(64, activation="relu", name="dense64_relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model_a")
    if model.count_params() != MODEL_A_EXPECTED_PARAMS:
        raise AssertionError(
            f"Model A should have {MODEL_A_EXPECTED_PARAMS} parameters, "
            f"but has {model.count_params()}"
        )
    if model.count_params() <= MODEL_A_PARAMETER_LIMIT:
        raise AssertionError("Model A is expected to be the over-100k standard CNN")
    return model