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
import keras
from keras import layers

MAX_PARAMS = 100_000


def build_model_b(input_shape=(28, 28, 1), num_classes=10):
    """Return the uncompiled Model B."""
    inputs = keras.Input(shape=input_shape)

    # First layer stays a standard conv: with 1 input channel a depthwise split saves nothing.
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(2, name="pool1")(x)                      # 28x28 -> 14x14

    # Depthwise 3x3 per channel, then 1x1 pointwise to mix channels.
    x = layers.SeparableConv2D(64, 3, padding="same", activation="relu", name="sepconv2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)                      # 14x14 -> 7x7

    x = layers.SeparableConv2D(128, 3, padding="same", activation="relu", name="sepconv3")(x)

    # Global average pooling instead of Flatten keeps the classifier head tiny.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="model_b")
    assert model.count_params() <= MAX_PARAMS, f"Model B has {model.count_params()} params"
    return model


def count_macs(model):
    """Multiply-accumulate operations for one image, for Conv2D, SeparableConv2D and Dense layers."""
    total = 0
    for layer in model.layers:
        if isinstance(layer, layers.SeparableConv2D):
            _, h, w, c_out = layer.output.shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            c_in = layer.input.shape[-1]
            total += h * w * (k * c_in + c_in * c_out)    # depthwise + pointwise
        elif isinstance(layer, layers.Conv2D):
            _, h, w, c_out = layer.output.shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            c_in = layer.input.shape[-1]
            total += h * w * k * c_in * c_out
        elif isinstance(layer, layers.Dense):
            total += layer.input.shape[-1] * layer.units
    return int(total)


if __name__ == "__main__":
    m = build_model_b()
    m.summary()
    print(f"MACs per image: {count_macs(m):,}")
