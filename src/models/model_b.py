"""Model B: lightweight CNN built from depthwise separable convolutions.

Same layout as Model A (conv 32 -> pool -> conv 64 -> pool -> conv 128 -> GAP ->
dense 64 -> dense 10), but the 2nd and 3rd convolutions are SeparableConv2D.
Only the convolution type differs, so A vs B isolates the effect of depthwise
separable convolutions.

Trainable parameters (3x3 kernels, bias per output channel):
    Conv2D 1->32            3*3*1*32 + 32             =    320
    SeparableConv2D 32->64  3*3*32 + 32*64 + 64       =  2,400
    SeparableConv2D 64->128 3*3*64 + 64*128 + 128     =  8,896
    Dense 128->64           128*64 + 64               =  8,256
    Dense 64->10            64*10 + 10                =    650
    Total                                             = 20,522  (limit: 100,000)

Activations: ReLU everywhere (a single compare, no exp(), int8-quantization
friendly). Softmax only on the output for the training loss; on-device
inference can take argmax of the logits instead.
"""
from tensorflow import keras
from tensorflow.keras import layers


MODEL_B_PARAMETER_LIMIT = 100_000
MODEL_B_EXPECTED_PARAMS = 20_522


def build_model_b(input_shape=(28, 28, 1), num_classes=10) -> keras.Model:
    """Build the depthwise-separable CNN used for Model B."""
    inputs = keras.Input(shape=input_shape, name="image")

    # First layer stays a standard conv: with 1 input channel a separable conv would learn a single
    # 3x3 filter and output 32 scaled copies of it. The standard conv costs only 320 params anyway.
    x = layers.Conv2D(32, 3, strides=1, padding="same", activation="relu", name="conv1_relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2, name="pool1")(x)                      # 28x28 -> 14x14

    # Depthwise 3x3 per channel, then 1x1 pointwise to mix channels.
    x = layers.SeparableConv2D(64, 3, strides=1, padding="same", activation="relu", name="sepconv2_relu")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool2")(x)                      # 14x14 -> 7x7

    x = layers.SeparableConv2D(128, 3, strides=1, padding="same", activation="relu", name="sepconv3_relu")(x)

    # Global average pooling instead of Flatten keeps the classifier head tiny.
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    # Same compact head as Model A.
    x = layers.Dense(64, activation="relu", name="dense64_relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model_b")
    if model.count_params() != MODEL_B_EXPECTED_PARAMS:
        raise AssertionError(
            f"Model B should have {MODEL_B_EXPECTED_PARAMS} parameters, "
            f"but has {model.count_params()}"
        )
    if model.count_params() > MODEL_B_PARAMETER_LIMIT:
        raise AssertionError(f"Model B exceeds the {MODEL_B_PARAMETER_LIMIT:,} parameter limit")
    return model
